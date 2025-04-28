import os
import pandas as pd
import joblib
from modules.model_guard import validate_all_models

# Fayl yo'llari
CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"
CHART_PATH = "data/chart.png"  # Agar grafik kerak bo'lsa

def update_model_and_predict():
    """
    Model va CSV fayl asosida bashorat qiladi va eng xavfsiz kataklarni aniqlaydi.
    """
    try:
        # CSV va model fayllari mavjudligini tekshirish
        if not os.path.exists(CSV_PATH) or not os.path.exists(MODEL_PATH):
            return "❌ CSV yoki Model fayli topilmadi."

        # CSV faylni o'qish
        data = pd.read_csv(CSV_PATH)
        if data.empty:
            return "❌ CSV fayli bo‘sh!"

        # CSV fayldan oxirgi satrni olish
        last_row = data.iloc[-1, :-1].values.reshape(1, -1)  # oxirgi ustun 'bombs_count' emas

        # Modelni yuklash
        models = joblib.load(MODEL_PATH)

        # Model bilan bashorat qilish
        results = validate_all_models(models, last_row)

        # Eng xavfsiz kataklarni aniqlash
        sorted_cells = sorted(results.items(), key=lambda x: x[1], reverse=True)
        safest_cells = [cell for cell, _ in sorted_cells[:7]]  # Eng xavfsiz 6-7 ta katak

        return safest_cells

    except Exception as e:
        return f"❗️ Xatolik yuz berdi: {str(e)}"

def write_bombs_and_update_model(bombs):
    """
    Yangi bombalar natijasini CSV faylga yozib saqlaydi va modelni yangilashga tayyorlaydi.
    """
    try:
        # Yangi qatorni yaratish
        row = [0] * 25  # 25 ta katak uchun default qiymat
        for b in bombs:
            if 1 <= b <= 25:
                row[b - 1] = 1  # Bombalar bor joyga 1 qo'yiladi
        row.append(len(bombs))  # Oxirgi ustunga bombalar sonini yozamiz

        # DataFrame yaratish
        columns = [f'cell_{i+1}' for i in range(25)] + ['bombs_count']
        new_df = pd.DataFrame([row], columns=columns)

        # CSV faylga qo'shish yoki yangi yaratish
        if os.path.exists(CSV_PATH):
            new_df.to_csv(CSV_PATH, mode='a', index=False, header=False)
        else:
            os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
            new_df.to_csv(CSV_PATH, index=False)

        print("✅ Bombalar CSV ga saqlandi.")

    except Exception as e:
        print(f"❗️ Bombalarni CSV ga saqlashda xatolik: {str(e)}")
