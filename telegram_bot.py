import telebot
import os
from modules.update_predict_mines import update_model_and_predict
from dotenv import load_dotenv
from modules.logger import log_info, log_error
from modules.csv_checker import check_csv_integrity

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = telebot.TeleBot(TOKEN)

# Boshqa komandalar uchun holat
@bot.message_handler(commands=['start'])
def start_handler(message):
    bot.send_message(message.chat.id, "Assalomu alaykum! Mines AI signal botiga xush kelibsiz.")

@bot.message_handler(commands=['signal'])
def signal_handler(message):
    try:
        if not check_csv_integrity():
            bot.send_message(message.chat.id, "❌ CSV fayl topilmadi yoki noto‘g‘ri tuzilgan.")
            return

        log_info("/signal komandasini ishga tushdi")
        result = update_model_and_predict()

        if isinstance(result, list):
            text = "\n".join(result)
        else:
            text = str(result)

        bot.send_photo(message.chat.id, open("chart.png", "rb"))
        bot.send_message(message.chat.id, f"Eng xavfsiz kataklar: {', '.join(result)}")

    except Exception as e:
        log_error(f"/signal xatolik: {str(e)}")
        bot.send_message(message.chat.id, f"Xatolik: {str(e)}")

@bot.message_handler(commands=['bombs'])
def bombs_handler(message):
    try:
        parts = message.text.split()
        bombs = [int(x) for x in parts[1:]]

        if len(bombs) != 3:
            bot.send_message(message.chat.id, "❌ Iltimos, aniq 3 ta bombani kiriting. Masalan: /bombs 5 10 15")
            return

        # CSV faylga qo‘shamiz
        from modules.csv_checker import append_bombs_to_csv
        append_bombs_to_csv(bombs)
        bot.send_message(message.chat.id, "Yozib olindi. Model yangilanmoqda...")
        bot.send_message(message.chat.id, "AI modeli yangilandi!")

    except Exception as e:
        log_error(f"/bombs xatolik: {str(e)}")
        bot.send_message(message.chat.id, f"Xatolik: {str(e)}")

bot.polling()
