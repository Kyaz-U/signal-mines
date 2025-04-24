import pandas as pd
import numpy as np
import joblib
import os

# Model mavjudligini tekshiramiz
if not os.path.exists("models/mines_rf_models.pkl"):
    os.system("python train_model.py")

# Modelni yuklaymiz
models = joblib.load("models/mines_rf_models.pkl")

# Xavfsiz kataklarni bashorat qilish funksiyasi
def predict_safest_cells(latest_games_df, top_k=6):
    if latest_games_df.shape[0] <= 5:
        print("Kamida 5 ta o‘yinga ehtiyoj bor.")
        return []

    df_to_use = latest_games_df.copy()

    # Faqat 24 ta kerakli ustunlar
    feature_columns = [f"cell_{i+1}" for i in range(24)]
    df_to_use = df_to_use[feature_columns]

    # Har safar oxirgi 5 ta satr o‘rtachasini emas, random 5 ta satrni olaylik:
    random_sample = df_to_use.sample(n=5, replace=False)
    avg_row = random_sample.mean().values.reshape(1, -1)

    predictions = {}
    for i in range(24):
        key = f"cell_{i+1}"
        if key in models:
            model = models[key]
            prob = model.predict_proba(avg_row)[0][1]
            predictions[key] = prob

    safest = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [cell for cell, prob in safest]
