import telebot
import os
from modules.update_predict_mines import update_model_and_predict
from dotenv import load_dotenv
from modules.logger import log_info, log_error
from modules.csv_checker import check_csv_integrity

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start_handler(message):
    bot.send_message(message.chat.id, "Assalomu alaykum! Mines AI signal botiga xush kelibsiz.")

@bot.message_handler(commands=['signal'])
def signal_handler(message):
    try:
        if not check_csv_integrity():
            bot.send_message(message.chat.id, "Xatolik: CSV fayl to‘liq emas yoki buzilgan.")
            return

        safe_cells = update_model_and_predict()
        msg = "Eng xavfsiz kataklar: " + ", ".join(safe_cells)
        bot.send_message(message.chat.id, msg)

        # Grafikni yuborish
        chart_path = "/tmp/chart.png"
        if os.path.exists(chart_path):
            with open(chart_path, "rb") as photo:
                bot.send_photo(message.chat.id, photo)
        else:
            bot.send_message(message.chat.id, "Xatolik: Grafik topilmadi.")

    except Exception as e:
        log_error(f"Signal xatoligi: {str(e)}")
        bot.send_message(message.chat.id, f"Xatolik: {str(e)}")

@bot.message_handler(commands=['help'])
def help_handler(message):
    bot.send_message(message.chat.id, "Foydalanish: /signal - Eng xavfsiz kataklarni olish.")

bot.polling()
