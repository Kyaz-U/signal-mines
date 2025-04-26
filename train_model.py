import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

# Fayl yo'llari
data_path = 'data/mines_data.csv'
models_dir = 'models'
model_path = os.path.join(models_dir, 'mines_rf_models-ultimate.pkl')

# Papkani yaratish (agar mavjud bo'lmasa)
os.makedirs(models_dir, exist_ok=True)

# Ma'lumotlarni yuklash
data = pd.read_csv(data_path)

# X va y ajratish
X = data.drop(columns=['bombs_count'])
y = data['bombs_count']

# Har bir katak (cell_1, cell_2, ...) uchun alohida model yaratamiz
models = {}
for i in range(1, 26):
    cell_col = f'cell_{i}'
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, data[cell_col])
    models[cell_col] = model

# Model faylini saqlash
with open(model_path, 'wb') as f:
    pickle.dump(models, f)

print("Barcha modellar muvaffaqiyatli saqlandi!")
