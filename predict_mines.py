# predict_mines.py
import pandas as pd
import joblib
import numpy as np

# Model faylini yuklaymiz
models = joblib.load("models/mines_rf_models.pkl")

# Yangi o'yin uchun input: avvalgi 30 ta o'yin statistikasi (yoki 0 lar bilan to'ldirilgan)
def predict_safest_cells(latest_games_df, top_k=6):
    """
    Kiruvchi: latest_games_df - oxirgi o'yinlar statistikasi (DataFrame shaklida, 25 ustun)
    Chiquvchi: xavfsiz bo'lish ehtimoli eng yuqori bo'lgan kataklar (list)
    """
    if latest_games_df.shape[0] < 5:
        print("Diqqat: Model yaxshi ishlashi uchun kamida 5 ta o'yin kerak.")

    avg_row = latest_games_df.mean().values.reshape(1, -1)  # o'rtacha holatni olamiz

    predictions = {}
    for i in range(25):
        model = models[f"cell_{i+1}"]
        prob = model.predict_proba(avg_row)[0][0]  # xavfsiz bo'lish ehtimoli
        predictions[f"cell_{i+1}"] = prob

    # Eng yuqori xavfsiz ehtimollik bo'yicha saralaymiz
    safest = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [cell for cell, prob in safest]
