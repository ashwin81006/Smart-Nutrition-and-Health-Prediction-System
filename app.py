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
    # Use default dataset or generate synthetic if not found
    if not os.path.exists("cleaned_food_data.csv"):
        print("⚠️ cleaned_food_data.csv not found. Generating synthetic dataset...")
        np.random.seed(42)
        n_samples = 5000
        grades = np.random.choice(['a','b','c','d','e'], n_samples, p=[0.2,0.3,0.25,0.15,0.1])
        df = pd.DataFrame({
            "product_name": [f"Product_{i}" for i in range(n_samples)],
            "nutrition_grade_fr": grades,
            "energy_100g": np.random.uniform(200, 2500, n_samples),
            "fat_100g": np.random.uniform(0, 50, n_samples),
            "saturated_fat_100g": np.random.uniform(0, 20, n_samples),
            "carbohydrates_100g": np.random.uniform(0, 80, n_samples),
            "sugars_100g": np.random.uniform(0, 60, n_samples),
            "fiber_100g": np.random.uniform(0, 20, n_samples),
            "proteins_100g": np.random.uniform(0, 30, n_samples),
            "salt_100g": np.random.uniform(0, 5, n_samples),
            "additives_n": np.random.poisson(2, n_samples),
            "is_processed": np.random.choice([0,1], n_samples, p=[0.6,0.4]),
            "ingredients_text": ["sugar, flour, oil"] * n_samples,
        })
        # Add grade-based correlations
        df.loc[df.nutrition_grade_fr=='a', 'sugars_100g'] *= 0.3
        df.loc[df.nutrition_grade_fr=='e', 'sugars_100g'] *= 2
    else:
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
    return df, model, le, acc, f1, report, cm, importances, X_test, y_test, y_pred

# Global variables (loaded once)
df, model, le, acc, f1, report, cm, importances, X_test, y_test, y_pred = load_and_train()
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

    avg_nutrients = df.groupby("nutrition_grade_fr")[["proteins_100g","fiber_100g","sugars_100g","fat_100g"]].mean().reset_index()
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

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        prod1 = request.form.get('product1')
        prod2 = request.form.get('product2')
        if prod1 and prod2 and prod1 in df['product_name'].values and prod2 in df['product_name'].values:
            r1 = df[df['product_name']==prod1].iloc[0]
            r2 = df[df['product_name']==prod2].iloc[0]
            g1 = le.inverse_transform(model.predict(pd.DataFrame([r1[FEATURE_COLS].fillna(0)])))[0]
            g2 = le.inverse_transform(model.predict(pd.DataFrame([r2[FEATURE_COLS].fillna(0)])))[0]
            # Comparison metrics
            metrics = [
                {"metric": "Fat (g/100g)", "val1": r1["fat_100g"], "val2": r2["fat_100g"], "lower_better": True},
                {"metric": "Sugars (g/100g)", "val1": r1["sugars_100g"], "val2": r2["sugars_100g"], "lower_better": True},
                {"metric": "Saturated Fat (g/100g)", "val1": r1["saturated_fat_100g"], "val2": r2["saturated_fat_100g"], "lower_better": True},
                {"metric": "Salt (g/100g)", "val1": r1["salt_100g"], "val2": r2["salt_100g"], "lower_better": True},
                {"metric": "Fiber (g/100g)", "val1": r1["fiber_100g"], "val2": r2["fiber_100g"], "lower_better": False},
                {"metric": "Protein (g/100g)", "val1": r1["proteins_100g"], "val2": r2["proteins_100g"], "lower_better": False},
            ]
            comparisons = []
            for m in metrics:
                if m["lower_better"]:
                    winner = prod1 if m["val1"] < m["val2"] else (prod2 if m["val2"] < m["val1"] else "Tie")
                else:
                    winner = prod1 if m["val1"] > m["val2"] else (prod2 if m["val2"] > m["val1"] else "Tie")
                comparisons.append({"Metric": m["metric"], prod1: f"{m['val1']:.1f}", prod2: f"{m['val2']:.1f}", "Winner": winner})
            # Bar chart
            nutrients_cmp = ["energy_100g","fat_100g","sugars_100g","fiber_100g","proteins_100g","salt_100g"]
            labels_cmp = ["Energy","Fat","Sugars","Fiber","Protein","Salt"]
            fig = go.Figure()
            fig.add_trace(go.Bar(name=prod1[:25], x=labels_cmp, y=[r1[n] for n in nutrients_cmp], marker_color=color_for_grade(g1)))
            fig.add_trace(go.Bar(name=prod2[:25], x=labels_cmp, y=[r2[n] for n in nutrients_cmp], marker_color=color_for_grade(g2)))
            fig.update_layout(barmode="group", title="Nutrient Comparison (per 100g)",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.2)", font_color="#f1f5f9")
            plot_bar = fig.to_html(full_html=False)
            # Radar
            cats = ["Energy","Fat","Sat.Fat","Sugars","Fiber","Protein"]
            maxs = [3992,100,50,100,30,100]
            cols_r = ["energy_100g","fat_100g","saturated_fat_100g","sugars_100g","fiber_100g","proteins_100g"]
            fig_radar = go.Figure()
            for row, grade, name in [(r1,g1,prod1),(r2,g2,prod2)]:
                pcts = [row[c]/m*100 for c,m in zip(cols_r,maxs)]
                fig_radar.add_trace(go.Scatterpolar(r=pcts+[pcts[0]], theta=cats+[cats[0]],
                                                    fill="toself", name=name[:25],
                                                    fillcolor=hex_to_rgba(color_for_grade(grade), alpha=0.2),
                                                    line_color=color_for_grade(grade)))
            fig_radar.update_layout(polar=dict(bgcolor="rgba(30,41,59,0.5)", radialaxis=dict(visible=True, range=[0,100]),
                                               angularaxis=dict(color="#94a3b8")),
                                    title="Radar Overlay", paper_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9")
            plot_radar = fig_radar.to_html(full_html=False)
            return render_template('compare.html', product_names=all_product_names, result=True,
                                   prod1=prod1, prod2=prod2, g1=g1, g2=g2, comparisons=comparisons,
                                   plot_bar=plot_bar, plot_radar=plot_radar, grade_labels=GRADE_LABELS)
    return render_template('compare.html', product_names=all_product_names, result=False)


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

if __name__ == '__main__':
    app.run(debug=True)