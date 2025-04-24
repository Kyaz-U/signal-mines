import os
import telebot
import pandas as pd
from dotenv import load_dotenv
from predict_mines import predict_safest_cells

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

CSV_PATH = "data/mines_data.csv"

def init_csv():
    if not os.path.exists(CSV_PATH):
        os.makedirs("data", exist_ok=True)
        with open(CSV_PATH, 'w') as f:
            header = ",".join([f"cell_{i+1}" for i in range(25)])
            f.write(header + "\n")

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Salom! Bombacha kataklarini yuboring: /bombs 7 12 23")

@bot.message_handler(commands=['bombs'])
def save_bombs(message):
    try:
        parts = message.text.split()[1:]
        bombs = set(int(p) for p in parts if p.isdigit() and 1 <= int(p) <= 25)
        row = [0 if (i+1) in bombs else 1 for i in range(25)]
        with open(CSV_PATH, 'a') as f:
            f.write(",".join(map(str, row)) + "\n")
        bot.send_message(message.chat.id, "Yozib olindi. Model yangilanmoqda...")

        # Avtomatik modelni qayta o'qitish
        os.system("python train_model.py")
        bot.send_message(message.chat.id, "AI modeli yangilandi!")
    except Exception as e:
        bot.send_message(message.chat.id, f"Xatolik: {str(e)}")

@bot.message_handler(commands=['signal'])
def send_prediction(message):
    try:
        df = pd.read_csv(CSV_PATH)
        if len(df) < 5:
            bot.send_message(message.chat.id, "Kamida 5 ta o'yin natijasi kerak!")
            return
        safest = predict_safest_cells(df)
        text = "Eng xavfsiz kataklar:\n" + ", ".join(safest)
        bot.send_message(message.chat.id, text)
        with open("chart.png", 'rb') as photo:
            bot.send_photo(message.chat.id, photo)
    except Exception as e:
        bot.send_message(message.chat.id, f"Xatolik: {str(e)}")

init_csv()
bot.polling()
