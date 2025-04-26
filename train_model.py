import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Fayl manzillari
CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models_ultimate.pkl"

def train_and_save_models():
    try:
        # CSV faylni yuklab olish
        data = pd.read_csv(CSV_PATH)

        # X va y ni ajratish
        X = data.drop(columns=["bombs_count"])

        models = {}

        # Har bir katak (cell_1, cell_2, ...) uchun alohida model yaratish
        for column in X.columns:
            target = X[column]
            features = X.drop(columns=[column])

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(features, target)
            models[column] = model

        # Modellarni saqlash
        joblib.dump(models, MODEL_PATH)
        print("\u2705 Barcha modellar muvaffaqiyatli o'qitildi va saqlandi!")

    except Exception as e:
        print(f"\u274C Xatolik yuz berdi: {str(e)}")

if __name__ == "__main__":
    train_and_save_models()
