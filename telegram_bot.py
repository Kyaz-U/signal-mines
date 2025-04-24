import os
if not os.path.exists("models/mines_rf_models.pkl"):
    os.system("python train_model.py")
import telebot
import pandas as pd
from predict_mines import predict_safest_cells

TOKEN = "YOUR_BOT_TOKEN_HERE"
bot = telebot.TeleBot(TOKEN)

def get_latest_games():
    df = pd.read_csv("data/mines_data.csv")
    return df.drop(columns=["bombs_count"]).tail(5)

@bot.message_handler(commands=['start'])
def welcome(message):
    bot.send_message(message.chat.id, "Assalomu alaykum! Wersal Mines Signal Botga xush kelibsiz. /signal ni bosing xavfsiz kataklarni olish uchun.")

@bot.message_handler(commands=['signal'])
def send_signal(message):
    try:
        data = get_latest_games()
        safest = predict_safest_cells(data)
        response = "Eng xavfsiz kataklar (AI model asosida):\n"
        response += ', '.join(safest)
        bot.send_message(message.chat.id, response)
    except Exception as e:
        bot.send_message(message.chat.id, f"Xatolik yuz berdi: {e}")

bot.polling()
