import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import os

# Fayl manzillari
CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"

def train_and_save_models():
    try:
        # CSV faylini yuklash
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"CSV fayl topilmadi: {CSV_PATH}")

        data = pd.read_csv(CSV_PATH)

        # X va y ni ajratish
        X = data.drop(columns=["bombs_count"])
        y = data["bombs_count"]

        # Modellar dictionary
        models = {}

        for column in X.columns:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X.drop(columns=[column]), X[column])  # Har bir katak uchun model
            models[column] = model

        # Modellarni saqlash
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(models, f)

        print("✅ Barcha modellar muvaffaqiyatli o'qitildi va saqlandi.")

    except Exception as e:
        print(f"❌ Xatolik yuz berdi: {str(e)}")

if __name__ == "__main__":
    train_and_save_models()
