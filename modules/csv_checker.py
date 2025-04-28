import os
import pandas as pd

CSV_PATH = "data/mines_data.csv"

def check_csv_integrity(csv_path=CSV_PATH, expected_columns=26):
    """
    CSV faylning to'g'riligini tekshiradi: har bir qator expected_columns ustunga ega bo'lishi kerak.
    """
    try:
        if not os.path.exists(csv_path):
            print(f"❌ CSV fayli topilmadi: {csv_path}")
            return False

        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"❌ CSV fayli bo‘sh: {csv_path}")
            return False

        for idx, row in df.iterrows():
            if len(row) != expected_columns:
                print(f"❗️ Xatolik: {idx+1}-qator ustunlar soni {len(row)} ta, {expected_columns} ta bo‘lishi kerak.")
                return False

        return True

    except Exception as e:
        print(f"❗️ CSV faylni tekshirishda xatolik: {str(e)}")
        return False
