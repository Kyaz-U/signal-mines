import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from modules.logger import log_info, log_error
from modules.model_guard import validate_all_models

MODEL_PATH = "models/mines_rf_models.pkl"

def load_models():
    if not os.path.exists(MODEL_PATH):
        log_error("❌ Model fayli topilmadi. Avval train_model.py ni ishga tushiring.")
        raise FileNotFoundError("Model mavjud emas")
    log_info("✅ Model fayli muvaffaqiyatli yuklandi.")
    return joblib.load(MODEL_PATH)

def predict_safest_cells(data, top_k=6):
    try:
        # Oxirgi 5 o'yindan o'rtacha qatordan foydalanamiz
        avg_row = data.tail(5).drop("bombs_count", axis=1).mean().values.reshape(1, -1)

        models = load_models()
        predictions = validate_all_models(models, avg_row)

        # Ehtimollar bo‘yicha tartiblab eng xavfsiz kataklarni tanlaymiz
        safest = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
        log_info(f"✅ Signal chiqarildi: {safest}")

        # Grafik chizamiz
        draw_chart(predictions)

        return [cell for cell, _ in safest]
    except Exception as e:
        log_error(f"❌ Signal chiqarishda xatolik: {str(e)}")
        return [f"Xatolik yuz berdi: {str(e)}"]

def draw_chart(predictions):
    try:
        keys = list(predictions.keys())
        vals = [v * 100 for v in predictions.values()]
        plt.figure(figsize=(12, 5))
        bars = plt.bar(keys, vals, color='green')
        plt.xticks(rotation=90)
        plt.ylabel("Xavfsizlik darajasi (%)")
        plt.title("AI Premium: 25 ta katak bo‘yicha xavfsizlik bashorati")
        plt.tight_layout()

        for bar, val in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha='center', va='bottom', fontsize=8)

        plt.savefig("chart.png")
        plt.close()
        log_info("✅ Grafik saqlandi: chart.png")
    except Exception as e:
        log_error(f"❌ Grafik chizishda xatolik: {str(e)}")
