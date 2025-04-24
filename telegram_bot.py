import os
import telebot
import pandas as pd
from predict_mines import predict_safest_cells
from dotenv import load_dotenv

# .env fayldan token va admin_id ni o‘qiymiz
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

# Model yo‘q bo‘lsa, avtomatik yaratamiz
if not os.path.exists("models/mines_rf_models.pkl"):
    os.system("python train_model.py")

bot = telebot.TeleBot(TOKEN)

# CSV'dan oxirgi o'yinlar statistikasi
def get_latest_games():
    df = pd.read_csv("data/mines_data.csv")
    return df.drop(columns=["bombs_count"]).tail(30)

@bot.message_handler(commands=['start'])
def welcome(message):
    bot.send_message(message.chat.id, "Assalomu alaykum! Wersal Mines Signal Botga xush kelibsiz.\n\n /signal ni bosing xavfsiz kataklarni olish uchun.")

@bot.message_handler(commands=['signal'])
def send_signal(message):
    try:
        data = get_latest_games()
        safest = predict_safest_cells(data, top_k=6)
        response = "Eng xavfsiz kataklar (AI model asosida):\n\n" + ", ".join(safest)
        bot.send_message(message.chat.id, response)
    except Exception as e:
        bot.send_message(message.chat.id, f"Xatolik yuz berdi: {e}")

bot.polling()
