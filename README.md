# 🧬 NutriScan AI — Smart Nutrition Analysis Platform

AI-Powered Smart Nutrition Analysis and Health Prediction System  
Built with **Streamlit · Scikit-learn · Plotly · Pandas**

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset in the same folder as the app
#    File expected: cleaned_food_data.csv

# 3. Run
streamlit run nutrition_app.py
```

App opens at → http://localhost:8501

---

## 📂 File Structure

```
📁 project/
├── nutrition_app.py          ← Main Streamlit app
├── requirements.txt          ← Python dependencies
├── cleaned_food_data.csv     ← Dataset (237K+ food products)
└── README.md
```

---

## 🗂️ App Pages

| Page | Description |
|---|---|
| 🏠 Dashboard | KPIs, grade distribution, scatter plots, heatmaps |
| 🔬 Predict & Analyze | Search or manually enter food → get grade + XAI + recommendations |
| 📊 Data Explorer | Filter, violin plots, correlation matrix, scatter matrix, raw table |
| ⚖️ Compare Products | Side-by-side nutrient comparison with radar overlay & verdict table |
| 🤖 Model Insights | Feature importance, confusion matrix, classification report |

---

## 🤖 ML Model

- **Algorithm**: Random Forest Classifier (150 trees, depth=18)
- **Features**: 14 nutritional attributes (energy, fat, sugars, fiber, protein, salt, additives, processing flags, derived ratios)
- **Target**: Nutri-Score grade (A / B / C / D / E)
- **Preprocessing**: StandardScaler + LabelEncoder
- **Train/Test**: 80% / 20% stratified split
- **Class balance**: `class_weight="balanced"`

---

## 🧠 Key Features

- **Real-time prediction** with probability bars per grade  
- **XAI explanations** — rule-based reasoning for every prediction  
- **Ingredient NLP** — additive detection from raw ingredient text  
- **Smart recommendations** — top 5 healthier alternatives from dataset  
- **Interactive visualizations** — radar charts, violin plots, heatmaps  
- **Product comparison** — head-to-head with winner verdict  
- **Custom CSV upload** — bring your own dataset  

---

## 📊 Dataset Columns Used

| Column | Description |
|---|---|
| `product_name` | Food product name |
| `ingredients_text` | Raw ingredient list |
| `additives_n` | Number of food additives |
| `nutrition_grade_fr` | Nutri-Score A–E (target) |
| `energy_100g` | Energy in kJ per 100g |
| `fat_100g` | Total fat g/100g |
| `saturated_fat_100g` | Saturated fat g/100g |
| `carbohydrates_100g` | Carbohydrates g/100g |
| `sugars_100g` | Sugars g/100g |
| `fiber_100g` | Dietary fibre g/100g |
| `proteins_100g` | Protein g/100g |
| `salt_100g` | Salt g/100g |
| `calorie_density` | Derived: energy/100 |
| `fat_ratio` | Derived: fat % of calories |
| `sugar_ratio` | Derived: sugar % of calories |
| `protein_ratio` | Derived: protein % of calories |
| `is_processed` | Binary: processed food flag |
