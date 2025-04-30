import os
import pandas as pd
import joblib

# Fayl yo'llari
CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"

def predict_safest_cells():
    try:
        if not os.path.exists(CSV_PATH):
            return "❌ CSV fayli topilmadi!"
        if not os.path.exists(MODEL_PATH):
            return "❌ Model fayli topilmadi!"

        data = pd.read_csv(CSV_PATH)
        if data.empty:
            return "❌ CSV fayli bo'sh!"

        # Eng oxirgi qator
        last_row = data.iloc[-1:, :-1].values.reshape(1, -1)

        # Modelni yuklash
        model = joblib.load(MODEL_PATH)
        probabilities = model.predict_proba(last_row)

        # Har bir katak uchun xavfsizlik ehtimolini olish
        safe_cells = [(i + 1, p[0]) for i, p in enumerate(probabilities) if p[0] > 0.5]

        if not safe_cells:
            return "❌ Xavfsiz kataklar topilmadi."

        # Eng xavfsiz 6-7 ta katak
        safe_cells = sorted(safe_cells, key=lambda x: -x[1])
        selected_cells = [str(cell[0]) for cell in safe_cells[:7]]

        return "✅ Xavfsiz kataklar: " + ", ".join(selected_cells)

    except Exception as e:
        return f"❌ Xatolik yuz berdi: {str(e)}"
