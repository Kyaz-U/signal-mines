import os
import telebot
import pandas as pd
from predict_mines import predict_safest_cells
from dotenv import load_dotenv

# .env faylni yuklaymiz
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

bot = telebot.TeleBot(TOKEN)

# CSV dan so‘nggi o‘yinlarni yuklash
def get_latest_games():
    df = pd.read_csv("data/mines_data.csv")
    return df.drop(columns=["bombs_count"]).tail(30)

# /start komandasi
@bot.message_handler(commands=['start'])
def welcome(message):
    bot.send_message(
        message.chat.id,
        "Assalomu alaykum! Wersal Mines Signal Botga xush kelibsiz!\n"
        "/signal buyrug‘ini yuboring — eng xavfsiz katakchalarni ko‘rish uchun."
    )

# /signal komandasi
@bot.message_handler(commands=['signal'])
def send_signal(message):
    try:
        data = get_latest_games()
        safest = predict_safest_cells(data, top_k=6)
        response = "Eng xavfsiz kataklar (AI model asosida):\n"
        response += ", ".join(safest)
        bot.send_message(message.chat.id, response)
    except Exception as e:
        bot.send_message(message.chat.id, f"Xatolik yuz berdi: {e}")

bot.polling()
