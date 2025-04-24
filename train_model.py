import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

model_path = "models/mines_rf_models.pkl"
csv_path = "data/mines_data.csv"

# CSV fayl borligini tekshiramiz
if not os.path.exists(csv_path):
    print("❌ data/mines_data.csv topilmadi. Iltimos, /bombs orqali ma’lumot kiriting.")
    exit()

# CSVni o‘qiymiz
df = pd.read_csv(csv_path)

# bomb_count ustunini chiqaramiz
X = df.drop(columns=["bombs_count"])
models = {}

# Har bir katak uchun alohida model
for i in range(25):
    col = f"cell_{i+1}"
    y = X[col]
    X_temp = X.drop(columns=[col])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_temp, y)
    models[col] = model

# Modellarni saqlaymiz
os.makedirs("models", exist_ok=True)
joblib.dump(models, model_path)
print("✅ Model saqlandi:", model_path)
