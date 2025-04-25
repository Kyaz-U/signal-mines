import pandas as pd
import os

def check_csv_integrity(csv_path="data/mines_data.csv", expected_columns=26):
    try:
        if not os.path.exists(csv_path):
            return False

        df = pd.read_csv(csv_path)
        for index, row in df.iterrows():
            if len(row) != expected_columns:
                print(f"Xatolik: {index+1}-qator {len(row)} ta ustunli, {expected_columns} kerak.")
                return False
        return True
    except Exception as e:
        print(f"CSV faylni tekshirishda xatolik: {e}")
        return False

def append_bombs_to_csv(bomb_cells, csv_path="data/mines_data.csv"):
    try:
        # Yangi qator tayyorlash
        row = [0] * 25
        for b in bomb_cells:
            if 1 <= b <= 25:
                row[b - 1] = 1
        row.append(3)  # bombs_count har doim 3

        # CSVga yozish
        columns = [f"cell_{i+1}" for i in range(25)] + ["bombs_count"]
        new_df = pd.DataFrame([row], columns=columns)

        if os.path.exists(csv_path):
            new_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            new_df.to_csv(csv_path, index=False)

    except Exception as e:
        print(f"CSVga yozishda xatolik: {e}")
