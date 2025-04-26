import telebot
import os
from dotenv import load_dotenv
from modules.update_predict_mines import update_model_and_predict, write_bombs_and_update_model
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
        # CSV faylni tekshirish
        if not check_csv_integrity():
            bot.send_message(message.chat.id, "Xatolik: CSV fayli to‘liq emas yoki noto‘g‘ri.")
            return

        # Modelni yangilash va bashorat qilish
        result = update_model_and_predict()

        if isinstance(result, str):
            bot.send_message(message.chat.id, f"Eng xavfsiz kataklar: {result}")
            return

        if isinstance(result, list) and len(result) > 0:
            msg = "Eng xavfsiz kataklar: " + ", ".join(result)
            bot.send_message(message.chat.id, msg)

            # Grafikni jo‘natish
            with open("data/chart.png", "rb") as photo:
                bot.send_photo(message.chat.id, photo)
        else:
            bot.send_message(message.chat.id, "Xatolik: Model yoki grafik topilmadi.")
    except Exception as e:
        bot.send_message(message.chat.id, f"Xatolik: {str(e)}")

@bot.message_handler(commands=['bombs'])
def bombs_handler(message):
    try:
        parts = message.text.strip().split()
        bombs = [int(p) for p in parts[1:] if p.isdigit() and 1 <= int(p) <= 25]
        if len(bombs) != 3:
            bot.send_message(message.chat.id, "Iltimos, aniq 3 ta bombali katak kiriting. Masalan: /bombs 5 10 17")
            return

        # Bombalarni CSVga yozish va modelni yangilash
        from modules.update_predict_mines import write_bombs_and_update_model
        write_bombs_and_update_model(bombs)
        bot.send_message(message.chat.id, "Yozib olindi. Model yangilanmoqda...")
        bot.send_message(message.chat.id, "AI modeli yangilandi!")

    except Exception as e:
        bot.send_message(message.chat.id, f"Xatolik: {str(e)}")

bot.polling()
