import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Fayl yo‘llari
CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models-ultimate.pkl"

# Ma’lumotni yuklash
data = pd.read_csv(CSV_PATH)

# X va y ajratish
features = [f"cell_{i}" for i in range(1, 26)]
X = data[features]
y = data["bombs_count"]

# Har bir cell uchun alohida model
models = {}

for i in range(1, 26):
    label = f"cell_{i}"
    y_cell = data[label]  # Bomb bor/yo‘qligi
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_cell)
    models[label] = model

# Modellarni saqlash
with open(MODEL_PATH, "wb") as f:
    pickle.dump(models, f)

print(f"✅ Barcha 25 ta model '{MODEL_PATH}' ga saqlandi.")
