import os
import telebot
import pandas as pd
from dotenv import load_dotenv
from predict_mines import predict_safest_cells

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

CSV_PATH = "data/mines_data.csv"

# CSV faylni yaratish
def init_csv():
    if not os.path.exists(CSV_PATH):
        os.makedirs("data", exist_ok=True)
        with open(CSV_PATH, "w") as f:
            header = [f"cell_{i+1}" for i in range(25)] + ["bombs_count"]
            f.write(",".join(header) + "\n")

# 3 ta bombali katak qabul qilish
@bot.message_handler(commands=["bombs"])
def save_bombs(message):
    parts = message.text.split()[1:]
    bombs = set(int(p) for p in parts if p.isdigit() and 1 <= int(p) <= 25)

    if len(bombs) != 3:
        bot.send_message(message.chat.id, "Iltimos, aynan 3 ta bombali katak kiriting (masalan: /bombs 5 12 21)")
        return

    row = [1 if (i+1) in bombs else 0 for i in range(25)]
    row.append(3)  # bombs_count ni ham yozamiz

    with open(CSV_PATH, "a") as f:
        f.write(",".join(map(str, row)) + "\n")

    bot.send_message(message.chat.id, "Yozib olindi. Model yangilanmoqda...")
    os.system("python train_model.py")
    bot.send_message(message.chat.id, "AI modeli yangilandi!")

# Signal chiqarish
@bot.message_handler(commands=["signal"])
def send_prediction(message):
    try:
        df = pd.read_csv(CSV_PATH)
        if len(df) < 5:
            bot.send_message(message.chat.id, "Kamida 5 ta o‘yin natijasi kerak!")
            return

        safest = predict_safest_cells(df, top_k=7)
        text = "Eng xavfsiz kataklar (AI model asosida):\n" + ", ".join(safest)
        bot.send_message(message.chat.id, text)

        # Grafikni yuborish
        with open("chart.png", "rb") as photo:
            bot.send_photo(message.chat.id, photo)

    except Exception as e:
        bot.send_message(message.chat.id, f"Xatolik: {str(e)}")

# Botni ishga tushurish
init_csv()
bot.polling()
