import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Fayl manzillari
CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"

def train_and_save_models():
    try:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"❌ CSV fayli topilmadi: {CSV_PATH}")

        data = pd.read_csv(CSV_PATH)
        if data.empty:
            raise ValueError("❌ CSV fayli bo'sh!")

        # X va y ni ajratish
        X = data.drop(columns=['bombs_count'])
        y = data['bombs_count']

        # Bitta umumiy model o'qitiladi (25 ta ustun asosida)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Modelni saqlash
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print("✅ Model muvaffaqiyatli o'qitildi va saqlandi.")

    except Exception as e:
        print(f"❌ Xatolik yuz berdi: {str(e)}")

if __name__ == "__main__":
    train_and_save_models()
