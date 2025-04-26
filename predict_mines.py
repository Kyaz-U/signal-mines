import pandas as pd
import pickle
import os
from modules.model_guard import validate_all_models

CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"

def predict_safest_cells():
    try:
        if not os.path.exists(CSV_PATH):
            return "CSV fayli topilmadi."

        if not os.path.exists(MODEL_PATH):
            return "Model fayli topilmadi."

        # CSV fayldan oxirgi ma'lumotni olish
        data = pd.read_csv(CSV_PATH)
        if data.empty:
            return "CSV fayl bo'sh."

        last_row = data.iloc[-1, :-1].values.reshape(1, -1)

        # Modelni yuklab olish
        with open(MODEL_PATH, "rb") as f:
            models = pickle.load(f)

        # Modellar bilan bashorat qilish
        results = validate_all_models(models, last_row)

        # Eng xavfsiz kataklarni tartiblash
        sorted_cells = sorted(results.items(), key=lambda x: x[1], reverse=True)
        safest_cells = [cell for cell, prob in sorted_cells[:7]]

        return safest_cells

    except Exception as e:
        return f"Xatolik yuz berdi: {str(e)}"
