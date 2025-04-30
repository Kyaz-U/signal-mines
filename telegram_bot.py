import os
import telebot
from dotenv import load_dotenv
from modules.predict_mines import predict_safest_cells
from modules.train_model import train_and_save_models
from modules.update_predict_mines import write_bombs_and_update_model
from modules.csv_checker import check_csv_integrity

# Muhit o'zgaruvchilarini yuklab olish
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = telebot.TeleBot(TOKEN)

# /start komandasi
@bot.message_handler(commands=['start'])
def start_handler(message):
    bot.send_message(message.chat.id, "Assalomu alaykum! Mines AI signal botiga xush kelibsiz.")

# /bombs komandasi (bombalarni qabul qilish)
@bot.message_handler(commands=['bombs'])
def bombs_handler(message):
    try:
        parts = message.text.strip().split()
        if len(parts) < 2:
            bot.send_message(message.chat.id, "❗️ Kamida 3 ta bombali katak kiriting. Masalan: /bombs 4 10 22")
            return

        bombs = [int(x) for x in parts[1:] if x.isdigit() and 1 <= int(x) <= 25]
        if len(bombs) < 3:
            bot.send_message(message.chat.id, "❗️ Kamida 3 ta to'g'ri formatdagi katak kiriting (1–25 orasida).")
            return

        response = write_bombs_and_update_model(bombs)
        bot.send_message(message.chat.id, response)

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Bombalarni saqlashda xatolik yuz berdi: {str(e)}")

# /signal komandasi (eng xavfsiz kataklarni yuborish)
@bot.message_handler(commands=['signal'])
def signal_handler(message):
    try:
        if not check_csv_integrity():
            bot.send_message(message.chat.id, "❌ CSV fayli to‘liq emas yoki noto‘g‘ri formatda.")
            return

        safe_cells = predict_safest_cells()

        if not isinstance(safe_cells, list) or not safe_cells:
            bot.send_message(message.chat.id, f"❌ Xatolik yoki natija topilmadi: {safe_cells}")
            return

        msg = f"✅ Eng xavfsiz kataklar: {', '.join(map(str, safe_cells))}"
        bot.send_message(message.chat.id, msg)

        # Grafik yuborish (agar mavjud bo‘lsa)
        chart_path = "data/chart.png"
        if os.path.exists(chart_path):
            with open(chart_path, 'rb') as photo:
                bot.send_photo(message.chat.id, photo)
        else:
            bot.send_message(message.chat.id, "⚠️ Grafik mavjud emas.")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Signal chiqarishda xatolik yuz berdi: {str(e)}")

if __name__ == "__main__":
    print("[INFO] Bot ishga tushdi!")
    bot.polling(none_stop=True)
