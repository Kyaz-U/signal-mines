import pandas as pd
import joblib
import os
from modules.logger import log_info, log_error
from modules.model_guard import validate_all_models
from modules.csv_checker import validate_csv, get_recent_data

MODEL_PATH = "models/mines_rf_models.pkl"
CSV_PATH = "data/mines_data.csv"


def load_models():
    if not os.path.exists(MODEL_PATH):
        log_error("Model fayli topilmadi. Avval train_model.py ni ishga tushiring.")
        raise FileNotFoundError("Model mavjud emas")
    return joblib.load(MODEL_PATH)


def predict_from_csv(csv_path=CSV_PATH, top_k=6):
    is_valid, msg = validate_csv(csv_path)
    if not is_valid:
        log_error(f"CSV xatolik: {msg}")
        raise ValueError(msg)

    try:
        data_vector = get_recent_data(csv_path, n=5)
        models = load_models()
        predictions = validate_all_models(models, data_vector)
        sorted_cells = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
        log_info(f"Premium signal: {sorted_cells}")
        return sorted_cells
    except Exception as e:
        log_error(f"Xatolik signal chiqarishda: {str(e)}")
        raise e
