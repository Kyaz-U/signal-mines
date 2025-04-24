import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

if not os.path.exists("models/mines_rf_models.pkl"):
    os.system("python train_model.py")

models = joblib.load("models/mines_rf_models.pkl")

def predict_safest_cells(data, top_k=7):
    avg_row = data.tail(5).mean().values.reshape(1, -1)
    predictions = {}
    for i in range(25):
        prob = models[f"cell_{i+1}"].predict_proba(avg_row)[0][1]
        predictions[f"cell_{i+1}"] = prob
    safest = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    draw_chart(predictions)
    return [cell for cell, _ in safest]

def draw_chart(predictions):
    keys = list(predictions.keys())
    vals = [v * 100 for v in predictions.values()]
    plt.figure(figsize=(10,4))
    plt.bar(keys, vals, color='green')
    plt.xticks(rotation=90)
    plt.title("Xavfsizlik darajasi (%)")
    plt.tight_layout()
    plt.savefig("chart.png")
    plt.close()
