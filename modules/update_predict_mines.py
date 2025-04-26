import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from modules.logger import log_info, log_error

CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"


def write_bombs_and_update_model(bomb_cells):
    try:
        df = pd.read_csv(CSV_PATH)

        # Yangi qator yaratish
        new_row = {f'cell_{i+1}': 0 for i in range(25)}
        for cell in bomb_cells:
            new_row[f'cell_{cell}'] = 1
        new_row['bombs_count'] = len(bomb_cells)

        # CSVga qo'shish
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)

        # Modelni qayta o'qitish
        X = df.drop(columns=["bombs_count"])
        models = {}

        for column in X.columns:
            y = X[column]
            features = X.drop(columns=[column])

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(features, y)
            models[column] = model

        joblib.dump(models, MODEL_PATH)

        return True, "✅ AI modeli yangilandi."

    except Exception as e:
        log_error(f"Modelni yangilashda xatolik: {str(e)}")
        return False, f"❌ Xatolik: {str(e)}"
