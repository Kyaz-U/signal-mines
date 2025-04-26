import pandas as pd
import random

# CSV faylga saqlash uchun sozlamalar
NUM_SAMPLES = 1000  # 1000 ta yozuv
NUM_CELLS = 25      # 25 ta katak

# Barcha katak ustunlari
columns = [f"cell_{i+1}" for i in range(NUM_CELLS)] + ["bombs_count"]

# Yangi ma'lumotlar
data = []

for _ in range(NUM_SAMPLES):
    # Har bir yozuv uchun 3 ta tasodifiy bombali katak tanlash
    bombs = random.sample(range(1, NUM_CELLS + 1), 3)
    
    # 0 bilan to'ldirilgan qator yaratish
    row = [0] * NUM_CELLS
    for bomb in bombs:
        row[bomb - 1] = 1  # bombali katakni 1 qilish
    
    row.append(3)  # bombs_count doim 3 ta
    data.append(row)

# DataFrame yaratish
df = pd.DataFrame(data, columns=columns)

# CSV faylga yozish
df.to_csv("data/mines_data.csv", index=False)

print("✅ 1000 ta yozuv bilan mines_data.csv yaratildi!")
