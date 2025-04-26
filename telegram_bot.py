import os
import telebot
from dotenv import load_dotenv
from modules.update_predict_mines import write_bombs_and_update_model
from predict_mines import load_model, prepare_input, predict_safe_cells
from modules.csv_checker import check_csv_integrity

# Tokenni yuklash
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

bot = telebot.TeleBot(TOKEN)

# /start komandasi
def start_handler(message):
    bot.send_message(message.chat.id, "Assalomu alaykum! Mines AI signal botiga xush kelibsiz.")

# /bombs komandasi
def bombs_handler(message):
    try:
        parts = message.text.strip().split()
        bombs = [int(p) for p in parts[1:] if p.isdigit() and 1 <= int(p) <= 25]

        if len(bombs) != 3:
            bot.send_message(message.chat.id, "Iltimos, aniq 3 ta bombali katak kiriting. Masalan: /bombs 5 10 17")
            return

        success, msg = write_bombs_and_update_model(bombs)
        if success:
            bot.send_message(message.chat.id, "✅ Bombalar saqlandi va model yangilandi.")
        else:
            bot.send_message(message.chat.id, f"❌ Xatolik: {msg}")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Xatolik yuz berdi: {str(e)}")

# /signal komandasi
def signal_handler(message):
    try:
        if not check_csv_integrity():
            bot.send_message(message.chat.id, "❌ CSV fayl xato yoki bo'sh.")
            return

        models = load_model()
        input_row = prepare_input()
        safe_cells = predict_safe_cells(models, input_row)

        msg = "Eng xavfsiz kataklar: " + ", ".join(safe_cells)
        bot.send_message(message.chat.id, msg)

        # Grafikni yuborish (agar kerak bo'lsa)
        if os.path.exists("data/chart.png"):
            with open("data/chart.png", "rb") as photo:
                bot.send_photo(message.chat.id, photo)

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Signal chiqarishda xatolik: {str(e)}")

# Komanda handlerlarini ulash
bot.message_handler(commands=['start'])(start_handler)
bot.message_handler(commands=['bombs'])(bombs_handler)
bot.message_handler(commands=['signal'])(signal_handler)

# Botni ishga tushirish
bot.polling()
