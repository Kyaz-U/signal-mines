import pandas as pd

def check_csv_integrity(csv_path: str, expected_columns: int = 26) -> bool:
    try:
        df = pd.read_csv(csv_path)
        for index, row in df.iterrows():
            if len(row) != expected_columns:
                print(f"Xatolik: {index+1}-qator {len(row)} ta ustunli, {expected_columns} kerak.")
                return False
        return True
    except Exception as e:
        print(f"CSV faylni tekshirishda xatolik: {e}")
        return False
