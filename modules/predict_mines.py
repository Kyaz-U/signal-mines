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

        # CSV faylni o'qish
        data = pd.read_csv(CSV_PATH)
        if data.empty:
            return "❌ CSV fayli bo'sh!"

        # Eng oxirgi qator
        last_row = data.iloc[-1, :-1].values.reshape(1, -1)  # Bombs_count emas, cell_1...cell_25

        # Modellarni yuklash
        models = joblib.load(MODEL_PATH)

        safe_cells = []

        for cell_name, model in models.items():
            probability = model.predict_proba(last_row)[0][1]  # 1-class ehtimoli
            if probability < 0.5:
                # xavfsiz deb baholanadigan cell
                cell_num = int(cell_name.split("_")[1])
                safe_cells.append((cell_num, probability))

        if not safe_cells:
            return "❌ Xavfsiz kataklar topilmadi."

        # Eng xavfsiz 6-7 ta katakni olish
        safe_cells = sorted(safe_cells, key=lambda x: x[1])
        selected_cells = [cell[0] for cell in safe_cells[:7]]  # 7 ta xavfsiz katak

        return selected_cells

    except Exception as e:
        return f"❌ Xatolik yuz berdi: {str(e)}"
