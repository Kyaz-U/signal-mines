import pandas as pd
import numpy as np
import joblib
import os

# Model mavjudligini tekshiramiz
if not os.path.exists("models/mines_rf_models.pkl"):
    os.system("python train_model.py")

# Modelni yuklaymiz
models = joblib.load("models/mines_rf_models.pkl")

# Eng xavfsiz kataklarni aniqlovchi funksiyani yozamiz
def predict_safest_cells(latest_games_df, top_k=6):
    if latest_games_df.shape[0] < 1:
        print("Kamida 1 ta o‘yinga ehtiyoj bor.")
        return []

    df_to_use = latest_games_df.tail(1).copy()

    if 'cell25' in df_to_use.columns:
        df_to_use = df_to_use.drop(columns=["cell25"])

    avg_row = df_to_use.values.reshape(1, -1)

    predictions = {}
    for i in range(25):
        key = f"cell_{i+1}"
        if key in models:
            model = models[key]
            prob = model.predict_proba(avg_row)[0][1]
            predictions[key] = prob

    safest = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [cell for cell, prob in safest]
