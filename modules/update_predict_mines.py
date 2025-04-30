import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from modules.train_model import train_and_save_models

# Fayl yo'llari
CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"
CHART_PATH = "data/chart.png"

def update_model_and_predict():
    try:
        if not os.path.exists(CSV_PATH):
            return "❌ CSV fayli topilmadi!"
        if not os.path.exists(MODEL_PATH):
            return "❌ Model fayli topilmadi!"

        data = pd.read_csv(CSV_PATH)
        if data.empty:
            return "❌ CSV fayli bo'sh!"

        last_row = data.iloc[-1:, :-1].values.reshape(1, -1)

        model = joblib.load(MODEL_PATH)
        probabilities = model.predict_proba(last_row)

        safe_cells = [(i + 1, p[0]) for i, p in enumerate(probabilities) if p[0] > 0.5]
        if not safe_cells:
            return "❌ Xavfsiz kataklar topilmadi."

        safe_cells = sorted(safe_cells, key=lambda x: -x[1])
        selected_cells = [str(cell[0]) for cell in safe_cells[:7]]

        # Grafik chizish
        cells = [f"cell_{i+1}" for i in range(25)]
        counts = data.iloc[-1:, :-1].values.flatten()

        plt.figure(figsize=(10, 5))
        plt.bar(cells, counts)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(CHART_PATH)
        plt.close()

        return "✅ Xavfsiz kataklar: " + ", ".join(selected_cells)

    except Exception as e:
        return f"❌ Xatolik yuz berdi: {str(e)}"

def write_bombs_and_update_model(bomb_cells):
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
        else:
            df = pd.DataFrame(columns=[f"cell_{i}" for i in range(1, 26)] + ["bombs_count"])

        new_row = [1 if i+1 not in bomb_cells else 0 for i in range(25)]
        new_row.append(len(bomb_cells))
        df.loc[len(df)] = new_row

        df.to_csv(CSV_PATH, index=False)

        train_and_save_models()
        return "✅ Bombalar saqlandi va model yangilandi."

    except Exception as e:
        return f"❌ Bombalarni saqlashda xatolik yuz berdi: {str(e)}"
