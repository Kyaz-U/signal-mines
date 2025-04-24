import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# CSV fayl yo‘li
data_path = "data/mines_data.csv"
df = pd.read_csv(data_path)

# X (input) ni ajratamiz
X = df.drop(columns=["bombs_count"])

# Har bir katak uchun model yaratamiz
models = {}
for i in range(25):
    y_cell = X.iloc[:, i]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X.drop(columns=[f"cell_{i+1}"]), y_cell)
    models[f"cell_{i+1}"] = model

# Papkani yaratamiz (agar yo‘q bo‘lsa)
os.makedirs("models", exist_ok=True)

# Modellarni saqlaymiz
joblib.dump(models, "models/mines_rf_models.pkl")
print("Model(lar) muvaffaqiyatli o‘qitildi va saqlandi.")
