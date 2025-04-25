import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from modules.model_guard import validate_all_models
from modules.logger import log_info, log_error

model_path = "models/mines_rf_models-3.pkl"

if not os.path.exists(model_path):
    os.system("python train_model.py")

models = joblib.load(model_path)

def predict_safest_cells(data, top_k=6):
    if "bombs_count" in data.columns:
        avg_row = data.tail(5).drop("bombs_count", axis=1).mean().values.reshape(1, -1)
    else:
        avg_row = data.tail(5).mean().values.reshape(1, -1)

    predictions = validate_all_models(models, avg_row)
    log_info(f"AI signal tahlil natijalari: {predictions}")

    draw_chart(predictions)

    sorted_cells = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return [cell for cell, _ in sorted_cells[:top_k]]

def draw_chart(predictions):
    keys = list(predictions.keys())
    vals = [v * 100 for v in predictions.values()]

    plt.figure(figsize=(10, 4))
    plt.bar(keys, vals)
    plt.xticks(rotation=90)
    plt.ylabel("Xavfsizlik (%)")
    plt.title("25 ta katak bo‘yicha bashorat")
    plt.tight_layout()
    plt.savefig("chart.png")
    plt.close()
