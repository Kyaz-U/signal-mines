import pandas as pd
import numpy as np
import joblib

# Model faylini yuklaymiz
models = joblib.load("models/mines_rf_models.pkl")

# Yangi o'yin uchun input: oxirgi o'yin statistikasi (DataFrame holatida)
def predict_safest_cells(latest_games_df, top_k=6):
    if latest_games_df.shape[0] < 5:
        print("Diqqat! Model yaxshi ishlashi uchun kamida 5 ta o'yin kerak.")
        return []

    # O'rtacha holatni olamiz
    avg_row = latest_games_df.mean().values.reshape(1, -1)

    # Har bir katak uchun xavfsiz ehtimolni hisoblaymiz
    predictions = {}
    for i in range(25):
        model = models[f"cell_{i+1}"]
        prob = model.predict_proba(avg_row)[0][1]  # 1 — xavfsiz bo'lish ehtimoli
        predictions[f"cell_{i+1}"] = prob

    # Eng yuqori xavfsiz ehtimollik bo'yicha saralaymiz
    safest = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [cell for cell, prob in safest]
