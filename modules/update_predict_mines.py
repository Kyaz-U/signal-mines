import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from modules.model_guard import validate_all_models
from modules.logger import log_info, log_error

MODEL_PATH = "models/mines_rf_models_ultimate.pkl"
CSV_PATH = "data/mines_data.csv"
CHART_PATH = "data/chart.png"


def draw_chart(predictions):
    try:
        keys = list(predictions.keys())
        vals = [v * 100 for v in predictions.values()]

        plt.figure(figsize=(12, 6))
        plt.bar(keys, vals)
        plt.xticks(rotation=90)
        plt.ylabel("Xavfsizlik (%)")
        plt.title("25 ta katak bo'yicha bashorat")
        plt.tight_layout()
        plt.savefig(CHART_PATH)
        plt.close()
    except Exception as e:
        log_error(f"Grafik chizishda xatolik: {str(e)}")


def update_model_and_predict():
    try:
        df = pd.read_csv(CSV_PATH)
        if len(df) < 1:
            raise Exception("Yetarli ma'lumot mavjud emas")

        avg_row = df.tail(1).values.reshape(1, -1)
        models = joblib.load(MODEL_PATH)

        predictions = validate_all_models(models, avg_row)
        log_info(f"AI signal tahlil natijalari: {predictions}")

        draw_chart(predictions)

        sorted_cells = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_k = 7
        return [cell for cell, _ in sorted_cells[:top_k]]

    except FileNotFoundError as e:
        log_error(f"Model yoki CSV fayl topilmadi: {str(e)}")
        return "Xatolik: Model yoki ma'lumot fayli topilmadi."
    except Exception as e:
        log_error(f"Bashoratda xatolik: {str(e)}")
        return "Xatolik: {str(e)}"


def write_bombs_and_update_model(bomb_cells):
    try:
        df = pd.read_csv(CSV_PATH)

        new_row = {f'cell_{i+1}': 0 for i in range(25)}
        for cell in bomb_cells:
            new_row[f'cell_{cell}'] = 1
        new_row['bombs_count'] = len(bomb_cells)

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)

        # Modelni yangilash
        X = df.drop(columns=["bombs_count"])
        models = {}

        for i in range(25):
            col = f"cell_{i+1}"
            y = X[col]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            models[col] = model

        joblib.dump(models, MODEL_PATH)

        return True, "AI modeli yangilandi"
    except Exception as e:
        log_error(f"Modelni yangilashda xatolik: {str(e)}")
        return False, f"Xatolik: {str(e)}"
