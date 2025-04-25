
import pandas as pd

def check_csv_format(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] != 26:
            return f"❌ Xatolik: ustunlar soni {df.shape[1]}, lekin 26 bo'lishi kerak!"
        if not all(f"cell_{i+1}" in df.columns for i in range(25)) or "bombs_count" not in df.columns:
            return "❌ Xatolik: kerakli ustun nomlari mavjud emas!"
        if df.tail(5).isnull().any().any():
            return "❌ Xatolik: Oxirgi 5 qatorda bo'sh (NaN) qiymatlar bor!"
        return "✅ CSV fayl formati to'g'ri!"
    except Exception as e:
        return f"❌ Xatolik: {str(e)}"
