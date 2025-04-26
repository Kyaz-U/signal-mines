import pandas as pd
import pickle
import os
from modules.model_guard import validate_all_models

CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"
CHART_PATH = "data/chart.png"

def update_model_and_predict():
    try:
        if not os.path.exists(CSV_PATH) or not os.path.exists(MODEL_PATH):
            return "CSV yoki Model fayli topilmadi."

        # CSV fayldan oxirgi satrni olish
        data = pd.read_csv(CSV_PATH)
        last_row = data.iloc[-1, :-1].values.reshape(1, -1)

        # Modelni yuklash
        with open(MODEL_PATH, 'rb') as f:
            models = pickle.load(f)

        # Model bilan bashorat qilish
        results = validate_all_models(models, last_row)

        # Natijalarni tartiblab, eng xavfsiz kataklarni aniqlash
        sorted_cells = sorted(results.items(), key=lambda x: x[1], reverse=True)
        safest_cells = [cell for cell, prob in sorted_cells[:7]]

        return safest_cells

    except Exception as e:
        return f"Xatolik yuz berdi: {str(e)}"

def write_bombs_and_update_model(bombs):
    try:
        # CSV faylga yangi bombalar yozish
        row = [0] * 25
        for b in bombs:
            if 1 <= b <= 25:
                row[b - 1] = 1
        row.append(3)  # Har doim bombalar soni = 3

        # Yangi qatorni CSVga qo'shish
        columns = [f'cell_{i+1}' for i in range(25)] + ['bombs_count']
        new_df = pd.DataFrame([row], columns=columns)

        if os.path.exists(CSV_PATH):
            new_df.to_csv(CSV_PATH, mode='a', index=False, header=False)
        else:
            new_df.to_csv(CSV_PATH, index=False)

        print("Bombalar CSVga saqlandi.")

    except Exception as e:
        print(f"Bombalarni yozishda xatolik: {str(e)}")
