import telebot
import pandas as pd
import os
from dotenv import load_dotenv
from predict_mines import predict_safest_cells

# .env faylni yuklaymiz
load_dotenv()

# Tokenni chaqiramiz
TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

# Ma'lumotlarni CSV dan olish
def get_latest_games():
    df = pd.read_csv("data/mines_data.csv")
    return df.drop(columns=["bombs_count"]).tail(30)

# /start komandasi
@bot.message_handler(commands=['start'])
def welcome(message):
    bot.send_message(message.chat.id, "Assalomu alaykum! Wersal Mines Signal Botiga xush kelibsiz.\n"
                                      "Xavfsiz kataklar ro‘yxatini olish uchun /signal ni bosing.")

# /signal komandasi
@bot.message_handler(commands=['signal'])
def send_signal(message):
    try:
        data = get_latest_games()
        safest = predict_safest_cells(data, top_k=5)
        response = "Eng xavfsiz kataklar:\n" + ", ".join(str(cell) for cell, _ in safest)
        bot.send_message(message.chat.id, response)
    except Exception as e:
        bot.send_message(message.chat.id, f"Xatolik yuz berdi: {e}")

# Botni ishga tushuramiz
bot.polling()
