import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from modules.model_guard import validate_all_models
from modules.logger import log_info, log_error

MODEL_PATH = "models/mines_rf_models-ultimate.pkl"
CSV_PATH = "data/mines_data.csv"
CHART_PATH = "data/chart.png"
TOP_K = 7  # eng xavfsiz 7 ta katak

def draw_chart(predictions):
    try:
        keys = list(predictions.keys())
        vals = [v * 100 for v in predictions.values()]
        plt.figure(figsize=(10, 4))
        plt.bar(keys, vals)
        plt.xticks(rotation=90)
        plt.ylabel("Xavfsizlik (%)")
        plt.title("25 ta katak bo‘yicha bashorat")
        plt.tight_layout()
        plt.savefig(CHART_PATH)
        plt.close()
    except Exception as e:
        log_error(f"Grafik chizishda xatolik: {e}")

def predict_safest_cells():
    # Modelni yuklash
    try:
        with open(MODEL_PATH, 'rb') as f:
            models = pickle.load(f)
    except Exception as e:
        log_error(f"Model faylini yuklashda xatolik: {e}")
        return "Xatolik: Model fayli topilmadi", []

    # CSVni o'qish
    try:
        df = pd.read_csv(CSV_PATH)
        avg_row = df.tail(5).mean(numeric_only=True).values.reshape(1, -1)
    except Exception as e:
        log_error(f"CSV o‘qishda xatolik: {e}")
        return "Xatolik: CSV faylida muammo bor", []

    # Bashorat qilish
    try:
        predictions = validate_all_models(models, avg_row)
        draw_chart(predictions)
        sorted_cells = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_cells = [cell for cell, _ in sorted_cells[:TOP_K]]
        return top_cells
    except Exception as e:
        log_error(f"AI model bashoratida xatolik: {e}")
        return "Xatolik: Bashoratda muammo", []
