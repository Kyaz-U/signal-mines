import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Fayl yo‘llari
csv_path = "data/mines_data.csv"
model_path = "models/mines_rf_models.pkl"

# Ma'lumotlar mavjudligini tekshiramiz
if not os.path.exists(csv_path):
    print("CSV fayl topilmadi. Iltimos, /bombs orqali ma'lumot kiriting.")
    exit()

# CSVni o'qish
print("CSV yuklanmoqda...")
df = pd.read_csv(csv_path)
X = df.drop(columns=["bombs_count"])

# Modellar
models = {}
print("Model o'qitish boshlandi...")
for i in range(1, 26):
    col = f"cell_{i}"
    y = X[col]
    X_temp = X.drop(columns=[col])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_temp, y)
    models[col] = model

# Modellarni saqlash
os.makedirs("models", exist_ok=True)
joblib.dump(models, model_path)
print("Model saqlandi: models/mines_rf_models.pkl")
