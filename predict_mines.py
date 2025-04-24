import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

model_path = "models/mines_rf_models.pkl"

if not os.path.exists(model_path):
    os.system("python train_model.py")

models = joblib.load(model_path)

def predict_safest_cells(data, top_k=6):
    if "bombs_count" in data.columns:
        avg_row = data.tail(5).drop("bombs_count", axis=1).mean().values.reshape(1, -1)
    else:
        avg_row = data.tail(5).mean().values.reshape(1, -1)

    predictions = {}
    for i in range(1, 26):
        col = f"cell_{i}"
        X_input = pd.DataFrame(avg_row).drop(columns=[i-1], axis=1)
        prob = models[col].predict_proba(X_input)[0][0]  # xavfsizlik ehtimoli
        predictions[col] = prob

    safest = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    draw_chart(predictions)
    return [cell for cell, _ in safest]

def draw_chart(predictions):
    keys = list(predictions.keys())
    vals = [v * 100 for v in predictions.values()]
    plt.figure(figsize=(10, 4))
    plt.bar(keys, vals)
    plt.xticks(rotation=90)
    plt.ylabel("%")
    plt.title("Xavfsizlik darajasi (%)")
    plt.tight_layout()
    plt.savefig("chart.png")
    plt.close()
