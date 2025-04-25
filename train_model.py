import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from modules.logger import log_info, log_error

model_path = "models/mines_rf_models-3.pkl"
csv_path = "data/mines_data.csv"

def train_models():
    try:
        if not os.path.exists(csv_path):
            log_error("❌ CSV fayl topilmadi.")
            print("❌ CSV fayl topilmadi.")
            return

        df = pd.read_csv(csv_path)

        if "bombs_count" not in df.columns or len(df) < 10:
            log_error("❌ Ma'lumotlar yetarli emas yoki 'bombs_count' yo'q.")
            print("❌ Ma'lumotlar yetarli emas.")
            return

        X = df.drop(columns=["bombs_count"])
        models = {}

        for i in range(25):
            col = f"cell_{i+1}"
            y = X[col]
            X_temp = X.drop(columns=[col])

            model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
            model.fit(X_temp, y)
            models[col] = model

        os.makedirs("models", exist_ok=True)
        joblib.dump(models, model_path)
        log_info("✅ Model muvaffaqiyatli o'qitildi va saqlandi.")
        print("✅ Model o'qitildi va saqlandi.")

    except Exception as e:
        log_error(f"❌ Model o'qitishda xatolik: {str(e)}")
        print(f"❌ Xatolik: {str(e)}")

if __name__ == "__main__":
    train_models()
