import pandas as pd
import matplotlib.pyplot as plt
from modules.model_guard import validate_all_models
from modules.logger import log_info, log_error

MODEL_PATH = "models/mines_rf_models-3.pkl"
CSV_PATH = "data/mines_data.csv"
CHART_PATH = "data/chart.png"

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
        log_error(f"Grafik chizishda xatolik: {str(e)}")

def predict_safest_cells():
    try:
        # CSV ni o‘qish
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            log_error("CSV fayl bo‘sh")
            return []

        avg_row = df.tail(5).mean(numeric_only=True).values.reshape(1, -1)

        # Modelni yuklash
        import joblib
        models = joblib.load(MODEL_PATH)

        # Bashorat qilish
        predictions = validate_all_models(models, avg_row)

        log_info(f"AI signal tahlil natijalari: {predictions}")

        if not predictions:
            log_error("Predictions bo‘sh qaytdi")
            return []

        draw_chart(predictions)

        sorted_cells = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [cell for cell, _ in sorted_cells[:6]]

    except Exception as e:
        log_error(f"Signal chiqarishda xatolik: {str(e)}")
        return []
