import pandas as pd
import joblib
from modules.model_guard import validate_all_models

CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"


def load_model(model_path=MODEL_PATH):
    try:
        models = joblib.load(model_path)
        return models
    except Exception as e:
        raise Exception(f"Modelni yuklashda xatolik: {str(e)}")


def prepare_input(csv_path=CSV_PATH):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise Exception("CSV faylda ma'lumot yo'q.")
        input_row = df.tail(1).drop(columns=["bombs_count"]).values.reshape(1, -1)
        return input_row
    except Exception as e:
        raise Exception(f"CSV faylni tayyorlashda xatolik: {str(e)}")


def predict_safe_cells(models, input_row, top_k=7):
    try:
        predictions = validate_all_models(models, input_row)
        sorted_cells = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        safe_cells = [cell for cell, _ in sorted_cells[:top_k]]
        return safe_cells
    except Exception as e:
        raise Exception(f"Bashoratda xatolik: {str(e)}")


if __name__ == "__main__":
    models = load_model()
    input_row = prepare_input()
    safe_cells = predict_safe_cells(models, input_row)
