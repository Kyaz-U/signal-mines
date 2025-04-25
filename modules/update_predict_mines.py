import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from modules.model_guard import validate_all_models
from modules.logger import log_info, log_error

model_path = "models/mines_rf_models-3.pkl"
csv_path = "data/mines_data.csv"

def draw_chart(predictions: dict):
    keys = list(predictions.keys())
    values = [v * 100 for v in predictions.values()]

    plt.figure(figsize=(12, 6))
    plt.bar(keys, values)
    plt.xticks(rotation=90)
    plt.title("25 ta katak bo‘yicha bashorat")
    plt.ylabel("Xavfsizlik darajasi (%)")
    plt.tight_layout()
    plt.savefig("chart.png")
    plt.close()

def update_model_and_predict(top_k: int = 6):
    try:
        if not os.path.exists(csv_path):
            log_error("❌ CSV fayl topilmadi.")
            return ["Xatolik: CSV fayl topilmadi"]

        df = pd.read_csv(csv_path)

        if len(df) < 10 or "bombs_count" not in df.columns:
            log_error("❌ CSV ma'lumotlar yetarli emas yoki 'bombs_count' ustuni yo‘q.")
            return ["Xatolik: Ma'lumotlar yetarli emas"]

        # Modelni yuklaymiz
        if not os.path.exists(model_path):
            log_error("❌ .pkl model fayl topilmadi.")
            return ["Xatolik: Model fayli topilmadi"]

        models = joblib.load(model_path)

        # So‘nggi 5 qatordan o‘rtacha qatormiz
        avg_row = df.tail(5).drop(columns=["bombs_count"]).mean().values.reshape(1, -1)

        # Modelga yuboramiz
        results = validate_all_models(models, avg_row)

        if not results:
            log_error("❌ Hech qanday natija qaytmadi.")
            return ["Xatolik: Bashorat natijasi yo‘q"]

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]

        draw_chart(results)
        safest_cells = [cell for cell, _ in sorted_results]
        log_info(f"✅ Eng xavfsiz kataklar: {safest_cells}")
        return safest_cells

    except Exception as e:
        log_error(f"❌ update_predict_mines.py xatolik: {str(e)}")
        return [f"Xatolik: {str(e)}"]
