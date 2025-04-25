import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from modules.logger import log_info, log_error

MODEL_PATH = "models/mines_rf_models.pkl"
CSV_PATH = "data/mines_data.csv"

def train_models():
    if not os.path.exists(CSV_PATH):
        log_error("CSV fayli topilmadi. /bombs orqali ma'lumot kiriting.")
        raise FileNotFoundError("CSV mavjud emas")

    df = pd.read_csv(CSV_PATH)
    if df.shape[1] != 26:
        log_error("CSV ustunlari noto‘g‘ri. 25 + 1 ustun kerak.")
        raise ValueError("CSV formati noto‘g‘ri")

    X_all = df.drop(columns=["bombs_count"])

    models = {}
    for i in range(25):
        col = f"cell_{i+1}"
        y = X_all[col]
        X = X_all.drop(columns=[col])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced_subsample'
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        models[col] = model
        log_info(f"Model [{col}] aniqligi: {round(acc * 100, 2)}%")

    os.makedirs("models", exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    log_info("✅ Barcha AI modellar o‘qitildi va models/mines_rf_models.pkl ga saqlandi.")

if __name__ == "__main__":
    train_models()
