import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

model_path = "models/mines_rf_models.pkl"

def predict_safest_cells(data: pd.DataFrame, top_k: int = 6):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model fayli topilmadi. Avval train_model.py ni ishga tushiring.")

    models = joblib.load(model_path)
    avg_row = data.tail(5).drop(columns=["bombs_count"]).mean()
    predictions = {}

    for i in range(1, 26):
        col = f"cell_{i}"
        input_features = avg_row.drop(labels=[col]).values.reshape(1, -1)
        prob = models[col].predict_proba(input_features)[0][0]  # xavfsiz ehtimoli
        predictions[col] = prob

    # Top xavfsiz kataklar
    safest = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Grafik chizish
    draw_chart(predictions)
    return [cell for cell, _ in safest]

def draw_chart(predictions: dict):
    cells = list(predictions.keys())
    probs = [round(predictions[cell] * 100, 2) for cell in cells]

    plt.figure(figsize=(12, 4))
    plt.bar(cells, probs)
    plt.xticks(rotation=90)
    plt.ylim(0, 100)
    plt.ylabel("Xavfsizlik darajasi (%)")
    plt.title("25 ta katak bo‘yicha bashorat")
    plt.tight_layout()
    plt.savefig("chart.png")
    plt.close()
