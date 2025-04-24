# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# CSV'dan o'yin ma'lumotlarini yuklab olamiz
df = pd.read_csv("data/mines_data.csv")

# X (input) ni ajratamiz
X = df.drop(columns=["bombs_count"])

# Har bir katak uchun alohida model tayyorlaymiz
models = {}
for i in range(25):
    y_cell = X.iloc[:, i]  # faqat shu katakni nishonlaymiz
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X.drop(columns=[f"cell_{i+1}"]), y_cell)
    models[f"cell_{i+1}"] = model

# Barcha modellarni saqlaymiz
joblib.dump(models, "models/mines_rf_models.pkl")
print("Model(lar) muvaffaqiyatli o'qitildi va saqlandi.")
