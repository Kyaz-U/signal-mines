import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

model_path = "models/mines_rf_models.pkl"
csv_path = "data/mines_data.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError("CSV topilmadi, iltimos, /bombs orqali ma'lumot kiriting.")

df = pd.read_csv(csv_path)
X = df.drop(columns=["bombs_count"])
models = {}
for i in range(25):
    col = f"cell_{i+1}"
    y = df[col]
    X_temp = X.drop(columns=[col])
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_temp, y)
    models[col] = model

os.makedirs("models", exist_ok=True)
joblib.dump(models, model_path)
