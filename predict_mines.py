import os
import pandas as pd
import joblib
from modules.model_guard import validate_all_models  # Agar validatsiya qilish kerak bo'lsa

# Fayl yo'llari
CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"

def predict_safest_cells():
    """
    Oxirgi olingan statistikaga asoslanib, eng xavfsiz katakchalarni bashorat qiladi.
    """
    try:
        # CSV fayl va model borligini tekshirish
        if not os.path.exists(CSV_PATH):
            return "❌ CSV fayli topilmadi!"

        if not os.path.exists(MODEL_PATH):
            return "❌ Model fayli topilmadi!"

        # CSV faylni o'qish
        data = pd.read_csv(CSV_PATH)
        if data.empty:
            return "❌ CSV fayli bo‘sh!"

        # Eng oxirgi qatorni olish
        last_row = data.iloc[-1].values.reshape(1, -1)

        # Modelni yuklash
        model = joblib.load(MODEL_PATH)

        # Bashorat qilish
        probabilities = model.predict_proba(last_row)[0]

        # 0.5 dan kichik ehtimollikdagi kataklar xavfsiz deb olinadi
        safe_cells = [i + 1 for i, prob in enumerate(probabilities) if prob < 0.5]

        # Agar xavfsiz kataklar topilmasa
        if not safe_cells:
            return "❗️ Xavfsiz kataklar topilmadi."

        return safe_cells

    except Exception as e:
        return f"❗️ Xatolik yuz berdi: {str(e)}"
