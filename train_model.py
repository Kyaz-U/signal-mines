import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

df = pd.read_csv("data/mines_data.csv")
models = {}

for i in range(25):
    X = df.drop(columns=[f"cell_{i+1}"])
    y = df[f"cell_{i+1}"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    models[f"cell_{i+1}"] = model

joblib.dump(models, "models/mines_rf_models.pkl")
print("AI model(lar) muvaffaqiyatli o'qitildi.")
