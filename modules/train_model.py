import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Fayl manzillari
CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"

def train_and_save_models():
    """
    CSV ma'lumotlar asosida barcha kataklar uchun RandomForest modellari yaratadi va saqlaydi.
    """
    try:
        # CSV fayl mavjudligini tekshirish
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"❌ CSV fayli topilmadi: {CSV_PATH}")

        # CSV faylni o'qish
        data = pd.read_csv(CSV_PATH)
        if data.empty:
            raise ValueError("❌ CSV fayli bo‘sh!")

        # X (input) va y (output) ni ajratish
        X = data.drop(columns=['bombs_count'])
        y = data['bombs_count']

        # Modellar dictionary'sini yaratish
        models = {}

        for column in X.columns:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X.drop(columns=[column]), X[column])  # Har bir katak uchun alohida model
            models[column] = model

        # Model faylini saqlash
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(models, MODEL_PATH)

        print("✅ Barcha modellar muvaffaqiyatli o'qitildi va saqlandi!")

    except Exception as e:
        print(f"❗️ Xatolik yuz berdi: {str(e)}")

if __name__ == "__main__":
    train_and_save_models()
