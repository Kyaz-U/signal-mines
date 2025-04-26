import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Fayl manzillari
CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"

def train_and_save_models():
    try:
        # CSV faylini yuklash
        data = pd.read_csv(CSV_PATH)

        # X va y ni ajratish
        X = data.drop(columns=["bombs_count"])

        models = {}

        for column in X.columns:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X.drop(columns=[column]), X[column])  # Har bir cell uchun model
            models[column] = model

        # Modelni saqlash
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(models, f)

        print("✅ Barcha modellar muvaffaqiyatli o'qitildi va saqlandi!")
    
    except Exception as e:
        print(f"❌ Xatolik yuz berdi: {str(e)}")

if __name__ == "__main__":
    train_and_save_models()
