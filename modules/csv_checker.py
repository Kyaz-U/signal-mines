import pandas as pd
import os

def validate_csv(csv_path):
    if not os.path.exists(csv_path):
        return False, "CSV fayl mavjud emas."

    df = pd.read_csv(csv_path)
    if df.shape[1] != 26:
        return False, f"CSV ustunlar soni noto'g'ri: {df.shape[1]} (kutilgan: 26)"

    if df.isnull().values.any():
        return False, "CSV faylda bo'sh qiymatlar mavjud."

    if df.shape[0] < 5:
        return False, "Kamida 5 ta satr bo'lishi kerak."

    return True, "CSV fayl to'g'ri."


def get_recent_data(csv_path, n=5):
    df = pd.read_csv(csv_path)
    return df.tail(n).drop("bombs_count", axis=1).mean().values.reshape(1, -1)
