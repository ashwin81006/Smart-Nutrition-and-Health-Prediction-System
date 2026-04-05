# app.py
import os
import re
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, session, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

app = Flask(__name__)
app.secret_key = 'nutrition-ai-secret-key'

# -------------------------------
# CONSTANTS & HELPERS
# -------------------------------
GRADE_COLORS = {"a": "#22c55e", "b": "#84cc16", "c": "#eab308", "d": "#f97316", "e": "#ef4444"}
GRADE_LABELS = {"a": "Excellent", "b": "Good", "c": "Average", "d": "Poor", "e": "Bad"}
FEATURE_COLS = [
    "energy_100g", "fat_100g", "saturated_fat_100g",
    "carbohydrates_100g", "sugars_100g", "fiber_100g",
    "proteins_100g", "salt_100g", "additives_n",
    "calorie_density", "fat_ratio", "sugar_ratio", "protein_ratio",
    "is_processed",
]

# Features used specifically for processed-food prediction
PROCESSED_FEATURE_COLS = [
    "energy_100g", "fat_100g", "saturated_fat_100g",
    "carbohydrates_100g", "sugars_100g", "fiber_100g",
    "proteins_100g", "salt_100g",
]

ADDITIVE_KEYWORDS = [
    "e1", "e2", "e3", "e4", "e5", "e6", "e9",
    "colour", "color", "flavor", "flavour", "preservative",
    "emulsifier", "stabiliser", "stabilizer", "sweetener",
    "antioxidant", "thickener", "acidity regulator",
    "modified starch", "artificial",
]


def parse_ingredients(text):
    if pd.isna(text) or not text:
        return []
    return [i.strip().lower() for i in re.split(r"[,;]", str(text)) if i.strip()]

def detect_additives_in_text(text):
    if pd.isna(text) or not text:
        return []
    found = []
    t = text.lower()
    for kw in ADDITIVE_KEYWORDS:
        if kw in t:
            found.append(kw.title())
    return list(set(found))

def explain_prediction(row, grade):
    reasons = []
    g = str(grade).lower()
    if row.get("sugars_100g", 0) > 30:
        reasons.append("🍬 Very high sugar content (>30g/100g)")
    elif row.get("sugars_100g", 0) > 15:
        reasons.append("🍬 High sugar content (>15g/100g)")
    if row.get("saturated_fat_100g", 0) > 10:
        reasons.append("🥓 High saturated fat (>10g/100g)")
    if row.get("salt_100g", 0) > 1.5:
        reasons.append("🧂 High salt content (>1.5g/100g)")
    if row.get("fiber_100g", 0) >= 5:
        reasons.append("🌾 Good fiber content (≥5g/100g) ✅")
    if row.get("proteins_100g", 0) >= 10:
        reasons.append("💪 High protein content (≥10g/100g) ✅")
    if row.get("additives_n", 0) > 5:
        reasons.append(f"⚗️ Many additives detected ({int(row['additives_n'])})")
    if row.get("is_processed", 0) == 1:
        reasons.append("🏭 Classified as processed food")
    if row.get("energy_100g", 0) > 2000:
        reasons.append("🔥 Very high caloric density (>2000 kJ/100g)")
    if row.get("fat_ratio", 0) > 0.5:
        reasons.append("⚠️ Fat makes up >50% of caloric composition")
    if not reasons:
        reasons.append("✅ Well-balanced nutritional profile across all dimensions")
    return reasons

def get_recommendations(grade, row, df):
    g = str(grade).lower()
    target_grades = {"d": ["a", "b"], "e": ["a", "b"], "c": ["a", "b"], "b": ["a"], "a": ["a"]}
    good = target_grades.get(g, ["a", "b"])
    sub = df[df["nutrition_grade_fr"].isin(good)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["_diff"] = abs(sub["energy_100g"] - row.get("energy_100g", 0))
    return sub.nsmallest(5, "_diff")[
        ["product_name", "nutrition_grade_fr", "energy_100g", "proteins_100g", "sugars_100g"]
    ].reset_index(drop=True)

def risk_level(grade):
    mapping = {"a": "Very Low", "b": "Low", "c": "Moderate", "d": "High", "e": "Very High"}
    return mapping.get(str(grade).lower(), "Unknown")

def additive_risk(n):
    if n == 0: return "None"
    if n <= 2: return "Low"
    if n <= 5: return "Moderate"
    return "High"

def color_for_grade(g):
    return GRADE_COLORS.get(str(g).lower(), "#94a3b8")

# -------------------------------
# DATA LOADING & MODEL TRAINING (cached globally)
# -------------------------------
def load_and_train():
    # Load real dataset only – no synthetic fallback
    if not os.path.exists("cleaned_food_data.csv"):
        raise FileNotFoundError(
            "❌ cleaned_food_data.csv not found! "
            "Please place your real dataset in the app directory."
        )
    
    df = pd.read_csv("cleaned_food_data.csv", low_memory=False)
    df["product_name"] = df["product_name"].astype(str).fillna("Unknown product")
    df["nutrition_grade_fr"] = df["nutrition_grade_fr"].str.lower().str.strip()
    df = df[df["nutrition_grade_fr"].isin(["a","b","c","d","e"])]

    # Derive features if not present
    for col in FEATURE_COLS:
        if col not in df.columns:
            if col == "calorie_density":
                df[col] = df["energy_100g"] / 100
            elif col == "fat_ratio":
                df[col] = df["fat_100g"] / (df["energy_100g"] / 37) if "energy_100g" in df else 0
            elif col == "sugar_ratio":
                df[col] = df["sugars_100g"] / (df["energy_100g"] / 17) if "energy_100g" in df else 0
            elif col == "protein_ratio":
                df[col] = df["proteins_100g"] / (df["energy_100g"] / 17) if "energy_100g" in df else 0
            else:
                df[col] = 0
    # Fill NaNs
    for col in FEATURE_COLS:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].fillna(df[col].median())
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df["nutrition_grade_fr"])
    X = df[FEATURE_COLS].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=4,
                                       class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    importances = model.named_steps["clf"].feature_importances_

    # ── Processed-food prediction models (Logistic Regression + Decision Tree) ──
    proc_df = df[PROCESSED_FEATURE_COLS + ["is_processed"]].dropna()
    Xp = proc_df[PROCESSED_FEATURE_COLS].values
    yp = proc_df["is_processed"].values

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        Xp, yp, test_size=0.2, random_state=42, stratify=yp
    )

    # Logistic Regression pipeline (with C=0.5 to reduce overfitting)
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=0.5, random_state=42)),
    ])
    lr_pipeline.fit(Xp_train, yp_train)
    lr_acc = accuracy_score(yp_test, lr_pipeline.predict(Xp_test))
    lr_f1  = f1_score(yp_test, lr_pipeline.predict(Xp_test), average="weighted")

    # Decision Tree pipeline (max_depth=5 to reduce overfitting)
    dt_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
    ])
    dt_pipeline.fit(Xp_train, yp_train)
    dt_acc = accuracy_score(yp_test, dt_pipeline.predict(Xp_test))
    dt_f1  = f1_score(yp_test, dt_pipeline.predict(Xp_test), average="weighted")
    dt_importances = dt_pipeline.named_steps["clf"].feature_importances_

    return (df, model, le, acc, f1, report, cm, importances, X_test, y_test, y_pred,
            lr_pipeline, lr_acc, lr_f1,
            dt_pipeline, dt_acc, dt_f1, dt_importances)

# Global variables (loaded once)
(df, model, le, acc, f1, report, cm, importances, X_test, y_test, y_pred,
 lr_proc_model, lr_proc_acc, lr_proc_f1,
 dt_proc_model, dt_proc_acc, dt_proc_f1, dt_proc_importances) = load_and_train()
all_product_names = sorted(df["product_name"].dropna().unique())
avg_energy = df["energy_100g"].mean()
avg_sugar = df["sugars_100g"].mean()
avg_fat = df["fat_100g"].mean()

# -------------------------------
# FLASK ROUTES
# -------------------------------
@app.route('/insights')
def insights():
    # Prepare plots as HTML
    grade_counts = df["nutrition_grade_fr"].value_counts().sort_index()
    fig1 = px.bar(x=grade_counts.index.str.upper(), y=grade_counts.values,
                  color=grade_counts.index, color_discrete_map=GRADE_COLORS,
                  labels={"x": "Nutri-Score Grade", "y": "# Products"},
                  title="Nutri-Score Distribution")
    fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.2)",
                       font_color="#f1f5f9")
    plot1_html = fig1.to_html(full_html=False)

    proc_counts = df["is_processed"].value_counts()
    fig2 = px.pie(names=["Natural / Minimally Processed", "Processed"],
                  values=[proc_counts.get(0,0), proc_counts.get(1,0)],
                  color_discrete_sequence=["#22c55e","#ef4444"], hole=0.55,
                  title="Processed vs Natural Foods")
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9")
    plot2_html = fig2.to_html(full_html=False)

    sample = df.sample(min(3000, len(df)), random_state=1)
    fig3 = px.scatter(sample, x="sugars_100g", y="fat_100g", color="nutrition_grade_fr",
                      color_discrete_map=GRADE_COLORS, opacity=0.45,
                      labels={"sugars_100g": "Sugars (g/100g)", "fat_100g": "Fat (g/100g)"},
                      title="Sugar vs Fat by Grade")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.2)", font_color="#f1f5f9")
    plot3_html = fig3.to_html(full_html=False)
    df_clean = df[
        (df["fiber_100g"] < 100) &
        (df["proteins_100g"] < 100) &
        (df["sugars_100g"] < 100) &
        (df["fat_100g"] < 100)
    ]
    avg_nutrients = df_clean.groupby("nutrition_grade_fr")[["proteins_100g","fiber_100g","sugars_100g","fat_100g"]].mean().reset_index()
    fig4 = go.Figure()
    nutrient_colors = {"proteins_100g":"#22c55e","fiber_100g":"#84cc16","sugars_100g":"#eab308","fat_100g":"#ef4444"}
    for nut, col in nutrient_colors.items():
        fig4.add_trace(go.Bar(x=avg_nutrients["nutrition_grade_fr"].str.upper(),
                              y=avg_nutrients[nut], name=nut.replace("_100g","").title(),
                              marker_color=col))
    fig4.update_layout(barmode="group", title="Average Nutrients by Grade",
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.2)", font_color="#f1f5f9")
    plot4_html = fig4.to_html(full_html=False)

    add_data = df.groupby("nutrition_grade_fr")["additives_n"].agg(["mean","median","max"]).reset_index()
    fig5 = go.Figure(data=go.Heatmap(z=add_data[["mean","median","max"]].values.T,
                                     x=add_data["nutrition_grade_fr"].str.upper(),
                                     y=["Mean","Median","Max"],
                                     colorscale=[[0,"#052e16"],[0.5,"#eab308"],[1,"#dc2626"]],
                                     text=np.round(add_data[["mean","median","max"]].values.T,1),
                                     texttemplate="%{text}", showscale=True))
    fig5.update_layout(title="Additive Count Statistics Across Grades",
                       paper_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9")
    plot5_html = fig5.to_html(full_html=False)

    kpis = {
        "Total Products": f"{len(df):,}",
        "Model Accuracy": f"{acc:.1%}",
        "Processed Foods": f"{df['is_processed'].mean():.1%}",
        "Avg Additives": f"{df['additives_n'].mean():.1f}",
        "Most Common Grade": df['nutrition_grade_fr'].value_counts().idxmax().upper()
    }
    return render_template('dashboard.html', kpis=kpis, plot1=plot1_html, plot2=plot2_html,
                           plot3=plot3_html, plot4=plot4_html, plot5=plot5_html)

def hex_to_rgba(hex_color, alpha=0.2):
    """Convert hex color (e.g. '#eab308') to rgba string with given alpha."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main Food Health Scanner page"""
    result = None
    if request.method == 'POST':
        if 'search_product' in request.form:
            # Search product by name
            product_name = request.form.get('product_name')
            matched = df[df["product_name"] == product_name]
            if not matched.empty:
                row = matched.iloc[0]
                pred_enc = model.predict(pd.DataFrame([row[FEATURE_COLS].fillna(0)]))[0]
                pred_grade = le.inverse_transform([pred_enc])[0]
                reasons = explain_prediction(row.to_dict(), pred_grade)
                recs = get_recommendations(pred_grade, row.to_dict(), df)
                # Ingredient analysis
                ingr_text = row.get("ingredients_text", "")
                ingredients = parse_ingredients(ingr_text)
                additives = detect_additives_in_text(ingr_text)
                # Processed detection
                is_processed = row.get("is_processed", 0)
                processed_label = "🏭 Highly processed" if is_processed else "🌿 Minimally processed"
                result = {
                    "type": "search",
                    "product_name": product_name,
                    "grade": pred_grade,
                    "grade_label": GRADE_LABELS.get(pred_grade, ""),
                    "health_status": "🟢 Excellent" if pred_grade == 'a' else 
                                     ("🟢 Good" if pred_grade == 'b' else
                                      ("🟡 Moderate" if pred_grade == 'c' else
                                       ("🔴 Poor" if pred_grade == 'd' else "🔴 Very Poor"))),
                    "reasons": reasons,
                    "recommendations": recs.to_dict('records') if not recs.empty else [],
                    "ingredient_count": len(ingredients),
                    "additives": additives,
                    "additive_count": len(additives),
                    "processed_label": processed_label,
                    "row": row.to_dict()
                }
        elif 'manual_input' in request.form:
            # Manual nutrition input
            energy = float(request.form['energy'])
            fat = float(request.form['fat'])
            sat_fat = float(request.form['sat_fat'])
            carbs = float(request.form['carbs'])
            sugars = float(request.form['sugars'])
            fiber = float(request.form['fiber'])
            protein = float(request.form['protein'])
            salt = float(request.form['salt'])
            additives_n = int(request.form['additives_n'])
            is_processed = 1 if request.form['is_processed'] == 'Yes' else 0
            ingr_text = request.form.get('ingr_text', '')
            # Derived features
            cal_density = energy / 100
            fat_ratio = fat / (energy / 37) if energy > 0 else 0
            sugar_ratio = sugars / (energy / 17) if energy > 0 else 0
            protein_ratio = protein / (energy / 17) if energy > 0 else 0
            input_row = {
                "energy_100g": energy, "fat_100g": fat, "saturated_fat_100g": sat_fat,
                "carbohydrates_100g": carbs, "sugars_100g": sugars, "fiber_100g": fiber,
                "proteins_100g": protein, "salt_100g": salt, "additives_n": additives_n,
                "calorie_density": cal_density, "fat_ratio": fat_ratio,
                "sugar_ratio": sugar_ratio, "protein_ratio": protein_ratio,
                "is_processed": is_processed,
            }
            X_input = pd.DataFrame([input_row])[FEATURE_COLS]
            pred_enc = model.predict(X_input)[0]
            pred_grade = le.inverse_transform([pred_enc])[0]
            reasons = explain_prediction(input_row, pred_grade)
            # Ingredient analysis
            ingredients = parse_ingredients(ingr_text) if ingr_text else []
            additives = detect_additives_in_text(ingr_text) if ingr_text else []
            processed_label = "🏭 Highly processed" if is_processed else "🌿 Minimally processed"
            result = {
                "type": "manual",
                "grade": pred_grade,
                "grade_label": GRADE_LABELS.get(pred_grade, ""),
                "health_status": "🟢 Excellent" if pred_grade == 'a' else 
                                 ("🟢 Good" if pred_grade == 'b' else
                                  ("🟡 Moderate" if pred_grade == 'c' else
                                   ("🔴 Poor" if pred_grade == 'd' else "🔴 Very Poor"))),
                "reasons": reasons,
                "recommendations": [],  # no recommendations for manual input (no product name to search)
                "ingredient_count": len(ingredients),
                "additives": additives,
                "additive_count": len(additives),
                "processed_label": processed_label,
                "input_row": input_row
            }
    return render_template('index.html', 
                         product_names=all_product_names,
                         result=result,
                         grade_labels=GRADE_LABELS,
                         risk_level=risk_level,
                         additive_risk=additive_risk,
                         color_for_grade=color_for_grade)

# Keep all other routes (dashboard, explorer, compare, model) unchanged
# ... (they remain as in previous version) ...  
def predict_future(series, steps=7):
    try:
        model = ARIMA(series, order=(3,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except Exception as e:
        print("ARIMA Error:", e)
        return None
    
@app.route('/explorer')
def explorer():
    # Filters from query string
    grades = request.args.getlist('grades') or ['A','B','C','D','E']
    processed = request.args.get('processed', 'All')
    max_energy = int(request.args.get('max_energy', 4000))
    grade_map = {g: g.lower() for g in grades}
    filt = df[df["nutrition_grade_fr"].isin([g.lower() for g in grades])]
    filt = filt[filt["energy_100g"] <= max_energy]
    if processed == "Processed":
        filt = filt[filt["is_processed"] == 1]
    elif processed == "Natural":
        filt = filt[filt["is_processed"] == 0]
    # Violin plots
    sample_violin = filt.sample(min(5000, len(filt)), random_state=1) if len(filt) > 0 else filt
    fig_v1 = px.violin(sample_violin, x="nutrition_grade_fr", y="sugars_100g",
                       color="nutrition_grade_fr", color_discrete_map=GRADE_COLORS,
                       box=True, title="Sugar Distribution by Grade")
    fig_v1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.2)", font_color="#f1f5f9")
    plot_v1 = fig_v1.to_html(full_html=False)
    fig_v2 = px.violin(sample_violin, x="nutrition_grade_fr", y="proteins_100g",
                       color="nutrition_grade_fr", color_discrete_map=GRADE_COLORS,
                       box=True, title="Protein Distribution by Grade")
    fig_v2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.2)", font_color="#f1f5f9")
    plot_v2 = fig_v2.to_html(full_html=False)
    # Correlation heatmap
    num_cols = ["energy_100g","fat_100g","saturated_fat_100g","sugars_100g","fiber_100g","proteins_100g","salt_100g","additives_n"]
    corr = filt[num_cols].corr()
    fig_corr = go.Figure(go.Heatmap(z=corr.values,
                                    x=[c.replace("_100g","").replace("_"," ").title() for c in corr.columns],
                                    y=[c.replace("_100g","").replace("_"," ").title() for c in corr.index],
                                    colorscale="RdYlGn", zmin=-1, zmax=1,
                                    text=np.round(corr.values,2), texttemplate="%{text}"))
    fig_corr.update_layout(title="Pearson Correlation", paper_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9")
    plot_corr = fig_corr.to_html(full_html=False)
    # Scatter matrix
    sample_scatter = filt.sample(min(2000, len(filt)), random_state=3) if len(filt) > 0 else filt
    fig_scat = px.scatter_matrix(sample_scatter, dimensions=["fat_100g","sugars_100g","fiber_100g","proteins_100g"],
                                 color="nutrition_grade_fr", color_discrete_map=GRADE_COLORS, opacity=0.4,
                                 title="Pairwise Nutrient Relationships")
    fig_scat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9")
    plot_scat = fig_scat.to_html(full_html=False)
    # Data table
    show_cols = ["product_name","nutrition_grade_fr","energy_100g","fat_100g","sugars_100g","proteins_100g","salt_100g","additives_n","is_processed"]
    table_data = filt[show_cols].head(200).to_dict('records')
    return render_template('explorer.html', total_products=len(filt), grades=grades, processed=processed,
                           max_energy=max_energy, plot_v1=plot_v1, plot_v2=plot_v2,
                           plot_corr=plot_corr, plot_scat=plot_scat, table_data=table_data)


from statsmodels.tsa.arima.model import ARIMA

def generate_time_series():
    """Generate synthetic user health data (for demo)"""
    np.random.seed(42)
    days = 30
    
    calories = np.random.randint(1800, 2800, days)
    
    weight = [70]
    for i in range(1, days):
        change = (calories[i] - 2200) / 7700  # realistic weight logic
        noise = np.random.normal(0, 0.1)
        weight.append(weight[-1] + change + noise)
    
    return pd.Series(weight)
from statsmodels.tsa.arima.model import ARIMA


@app.route('/forecast', methods=['GET','POST'])
def forecast():
    plot_html = None
    insight = None
    dataset_insight = None

    if request.method == 'POST':
        try:
            file = request.files['file']
            df_user = pd.read_csv(file)

            if 'weight' not in df_user.columns:
                return "CSV must contain 'weight' column"

            series = df_user['weight']

            # 🔥 ARIMA prediction
            forecast_vals = predict_future(series)

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=series, name="Past Weight"))

            if forecast_vals is not None:
                future_x = list(range(len(series), len(series)+len(forecast_vals)))

                fig.add_trace(go.Scatter(
                    x=future_x,
                    y=forecast_vals,
                    name="Predicted Weight"
                ))

                # 🔥 USER TREND
                trend = forecast_vals.mean() - series.mean()

                # 🔥 DATASET BASED LOGIC
                if 'calories' in df_user.columns:
                    user_avg_cal = df_user['calories'].mean()
                else:
                    user_avg_cal = 2000

                # compare with dataset baseline
                if user_avg_cal > avg_energy * 10:
                    dataset_insight = "⚠️ Your calorie intake is higher than typical food patterns."
                else:
                    dataset_insight = "✅ Your calorie intake is within a normal range."

                # combine both
                if trend > 0.2:
                    insight = "⚠️ Your weight is likely to increase."
                elif trend < -0.2:
                    insight = "✅ Your weight may decrease."
                else:
                    insight = "⚖️ Your weight is stable."

            fig.update_layout(
                title="📈 Personalized Weight Prediction",
                xaxis_title="Days",
                yaxis_title="Weight (kg)"
            )

            plot_html = fig.to_html(full_html=False)

        except Exception as e:
            print("Error:", e)

    return render_template(
        "forecast.html",
        plot=plot_html,
        insight=insight,
        dataset_insight=dataset_insight
    )
@app.route('/model')
def model_insights():
    # Feature importance
    fi_df = pd.DataFrame({"Feature": [f.replace("_100g","").replace("_"," ").title() for f in FEATURE_COLS],
                          "Importance": importances}).sort_values("Importance", ascending=True)
    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", color="Importance",
                    color_continuous_scale=["#1e293b","#22c55e"], title="Feature Importance (Random Forest)")
    fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.2)", font_color="#f1f5f9", coloraxis_showscale=False)
    plot_fi = fig_fi.to_html(full_html=False)
    # Confusion matrix
    labels = [g.upper() for g in le.classes_]
    fig_cm = go.Figure(go.Heatmap(z=cm, x=labels, y=labels,
                                  colorscale=[[0,"#0f172a"],[0.5,"#1d4ed8"],[1,"#22c55e"]],
                                  text=cm.astype(str), texttemplate="%{text}", showscale=True))
    fig_cm.update_layout(title="Confusion Matrix (Test Set)", xaxis_title="Predicted", yaxis_title="Actual",
                         paper_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9")
    plot_cm = fig_cm.to_html(full_html=False)
    # Classification report as table
    report_df = pd.DataFrame(report).T
    report_df = report_df.drop(index=["accuracy","macro avg","weighted avg"], errors="ignore")
    report_df = report_df.rename(index=str.upper)
    class_report = report_df[["precision","recall","f1-score"]].round(3).to_dict('records')
    # Prediction vs actual scatter
    pred_grades = le.inverse_transform(y_pred)
    true_grades = le.inverse_transform(y_test)
    agg = pd.DataFrame({"Actual": true_grades, "Predicted": pred_grades}).groupby(["Actual","Predicted"]).size().reset_index(name="Count")
    fig_scatter = px.scatter(agg, x="Actual", y="Predicted", size="Count", color="Count", color_continuous_scale="Viridis",
                             title="Prediction Agreement (bubble size = count)")
    fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9")
    plot_scatter = fig_scatter.to_html(full_html=False)
    return render_template('model.html', acc=acc, f1=f1, n_trees=model.named_steps["clf"].n_estimators,
                           plot_fi=plot_fi, plot_cm=plot_cm, class_report=class_report,
                           plot_scatter=plot_scatter, feature_count=len(FEATURE_COLS))


# -------------------------------------------------------
# PROCESSED FOOD PREDICTION  (Logistic Regression + DT)
# -------------------------------------------------------
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    result = None

    # Build feature-importance bar chart for DT (always shown)
    fi_df = pd.DataFrame({
        "Feature": [f.replace("_100g","").replace("_"," ").title() for f in PROCESSED_FEATURE_COLS],
        "Importance": dt_proc_importances
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=[[0,"#1e3a5f"],[0.5,"#6366f1"],[1,"#a855f7"]],
        title="Decision Tree – Feature Importance (Processed Food Prediction)"
    )
    fig_fi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,0.2)",
        font_color="#f1f5f9",
        coloraxis_showscale=False
    )
    plot_fi = fig_fi.to_html(full_html=False)

    if request.method == 'POST':
        try:
            energy   = float(request.form.get('energy',   0))
            fat      = float(request.form.get('fat',      0))
            sat_fat  = float(request.form.get('sat_fat',  0))
            carbs    = float(request.form.get('carbs',    0))
            sugars   = float(request.form.get('sugars',   0))
            fiber    = float(request.form.get('fiber',    0))
            protein  = float(request.form.get('protein',  0))
            salt     = float(request.form.get('salt',     0))
            additives_n = float(request.form.get('additives_n', 0))

            input_vec = np.array([[energy, fat, sat_fat, carbs, sugars,
                                   fiber, protein, salt, additives_n]])

            # ── Logistic Regression ──
            lr_pred  = int(lr_proc_model.predict(input_vec)[0])
            lr_proba = lr_proc_model.predict_proba(input_vec)[0]
            lr_conf  = float(lr_proba[lr_pred]) * 100

            # ── Decision Tree ──
            dt_pred  = int(dt_proc_model.predict(input_vec)[0])
            dt_proba = dt_proc_model.predict_proba(input_vec)[0]
            dt_conf  = float(dt_proba[dt_pred]) * 100

            # ── Explanation reasons (NEW) ──
            reasons = []
            if sugars > 15:
                reasons.append("🍬 High sugar content (>15g) – often indicates processing")
            if additives_n > 3:
                reasons.append("⚗️ More than 3 additives – typical for industrial foods")
            if fiber < 2:
                reasons.append("🌾 Low fiber (<2g) – suggests refined/processed ingredients")
            if fat > 20:
                reasons.append("🥑 High fat (>20g) – common in processed products")
            if not reasons:
                reasons.append("✅ Based on these values, the food looks minimally processed")

            # ── Weighted ensemble (60% LR + 40% DT) ──
            lr_proc_prob = lr_proba[1]   # probability of class 1 (Processed)
            dt_proc_prob = dt_proba[1]   # probability of class 1 (Processed)
            ensemble_score = lr_proc_prob * 0.6 + dt_proc_prob * 0.4
            ensemble_threshold = 0.5

            if ensemble_score >= ensemble_threshold:
                ensemble = "Processed"
                ensemble_icon = "🏭"
                ensemble_color = "#ef4444"
            else:
                ensemble = "Natural / Minimally Processed"
                ensemble_icon = "🌿"
                ensemble_color = "#22c55e"

            result = {
                "lr_label":   "Processed" if lr_pred == 1 else "Natural",
                "lr_conf":    round(lr_conf, 1),
                "lr_acc":     round(lr_proc_acc * 100, 1),
                "lr_f1":      round(lr_proc_f1, 3),
                "lr_color":   "#ef4444" if lr_pred == 1 else "#22c55e",
                "dt_label":   "Processed" if dt_pred == 1 else "Natural",
                "dt_conf":    round(dt_conf, 1),
                "dt_acc":     round(dt_proc_acc * 100, 1),
                "dt_f1":      round(dt_proc_f1, 3),
                "dt_color":   "#ef4444" if dt_pred == 1 else "#22c55e",
                "ensemble":        ensemble,
                "ensemble_icon":   ensemble_icon,
                "ensemble_color":  ensemble_color,
                "reasons":         reasons,          # <-- explanation added
                # Input echo
                "energy":      energy,  "fat": fat, "sat_fat": sat_fat,
                "carbs":       carbs,   "sugars": sugars, "fiber": fiber,
                "protein":     protein, "salt": salt, "additives_n": int(additives_n),
            }
        except Exception as exc:
            result = {"error": str(exc)}

    return render_template('prediction.html',
                           plot_fi=plot_fi,
                           result=result,
                           lr_acc=round(lr_proc_acc*100, 1),
                           lr_f1=round(lr_proc_f1, 3),
                           dt_acc=round(dt_proc_acc*100, 1),
                           dt_f1=round(dt_proc_f1, 3))


if __name__ == '__main__':
    app.run(debug=True)