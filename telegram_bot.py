import telebot
import pandas as pd
from predict_mines import predict_safest_cells
from dotenv import load_dotenv
import os

# .env fayldan tokenni yuklaymiz
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

# Oxirgi 30 ta o‘yinni yuklaymiz
def get_latest_games():
    df = pd.read_csv("data/mines_data.csv")
    return df.drop(columns=["bombs_count"]).tail(30)

@bot.message_handler(commands=['start'])
def welcome(message):
    bot.send_message(message.chat.id, "Assalomu alaykum!\nWersal Mines Signal Botga xush kelibsiz.\n\n/signal ni bosing xavfsiz kataklar ro‘yxatini olish uchun.")

@bot.message_handler(commands=['signal'])
def send_signal(message):
    try:
        data = get_latest_games()
        safest = predict_safest_cells(data, top_k=6)
        response = "AI bashorati bo‘yicha eng xavfsiz kataklar:\n\n" + ", ".join(safest)
        bot.send_message(message.chat.id, response)
    except Exception as e:
        bot.send_message(message.chat.id, f"Xatolik yuz berdi:\n{e}")

bot.polling()
