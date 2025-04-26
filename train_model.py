import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import os

CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"

def train_and_save_models():
    try:
        if not os.path.exists(CSV_PATH):
            print("❌ CSV fayli topilmadi.")
            return

        # CSV faylni o'qish
        data = pd.read_csv(CSV_PATH)

        if data.empty:
            print("❌ CSV fayli bo'sh.")
            return

        X = data.drop(columns=["bombs_count"])
        y = data["bombs_count"]

        # Har bir katak uchun alohida model yaratamiz
        models = {}

        for column in X.columns:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X.drop(columns=[column]), X[column])
            models[column] = model

        # Modellarni saqlash
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(models, f)

        print("✅ Barcha modelllar muvaffaqiyatli o'qitildi va saqlandi.")

    except Exception as e:
        print(f"❌ Model o'qitish jarayonida xatolik: {str(e)}")

if __name__ == "__main__":
    train_and_save_models()
