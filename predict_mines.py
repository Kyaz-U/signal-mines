import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

def draw_chart(predictions):
    plt.figure(figsize=(10, 4))
    sorted_preds = sorted(predictions.items(), key=lambda x: x[0])
    keys, vals = zip(*[(k, round(v * 100, 2)) for k, v in sorted_preds])
    plt.bar(keys, vals)
    plt.title("Xavfsizlik darajasi (%)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("chart.png")

def predict_safest_cells(data, top_k=6):
    X = data.drop(columns=["bombs_count"])
    avg_row = X.tail(5).mean().values.reshape(1, -1)
    models = joblib.load("models/mines_rf_models.pkl")
    predictions = {
        f"cell_{i+1}": models[f"cell_{i+1}"].predict_proba(avg_row)[0][1]
        for i in range(25)
    }
    draw_chart(predictions)
    safest = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [cell for cell, _ in safest]
