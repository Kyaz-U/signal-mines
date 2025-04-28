import os
import telebot
from dotenv import load_dotenv
from modules.predict_mines import predict_safest_cells
from modules.train_model import train_and_save_models
from modules.logger import log_info, log_error
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
        bombs = [int(p) for p in parts[1:] if p.isdigit() and 1 <= int(p) <= 25]

        if not bombs or len(bombs) < 3:
            bot.send_message(message.chat.id, "❗️ Iltimos, kamida 3 ta bombali katak kiriting. Masalan: /bombs 5 10 17")
            return

        # Bombalarni CSV ga yozish va modelni yangilash
        from modules.update_predict_mines import write_bombs_and_update_model
        write_bombs_and_update_model(bombs)

        bot.send_message(message.chat.id, "✅ Bombalar saqlandi va model yangilandi.")

    except Exception as e:
        log_error(f"/bombs komandasi xatolik: {str(e)}")
        bot.send_message(message.chat.id, f"❌ Bombalarni saqlashda xatolik yuz berdi: {str(e)}")

# /signal komandasi (eng xavfsiz kataklarni yuborish)
@bot.message_handler(commands=['signal'])
def signal_handler(message):
    try:
        if not check_csv_integrity():
            bot.send_message(message.chat.id, "❌ CSV fayli to'liq emas yoki noto'g'ri.")
            return

        safe_cells = predict_safest_cells()

        if isinstance(safe_cells, str):
            bot.send_message(message.chat.id, f"❌ Xatolik: {safe_cells}")
            return

        if isinstance(safe_cells, list) and len(safe_cells) > 0:
            msg = f"✅ Eng xavfsiz kataklar: {', '.join(map(str, safe_cells))}"
            bot.send_message(message.chat.id, msg)

            # Grafik yuborish (agar mavjud bo'lsa)
            chart_path = 'data/chart.png'
            if os.path.exists(chart_path):
                with open(chart_path, 'rb') as photo:
                    bot.send_photo(message.chat.id, photo)
            else:
                bot.send_message(message.chat.id, "ℹ️ Grafik mavjud emas.")
        else:
            bot.send_message(message.chat.id, "❗️ Xavfsiz kataklar topilmadi.")

    except Exception as e:
        log_error(f"/signal komandasi xatolik: {str(e)}")
        bot.send_message(message.chat.id, f"❌ Signal chiqarishda xatolik yuz berdi: {str(e)}")

if __name__ == "__main__":
    log_info("Bot ishga tushdi!")
    bot.polling(none_stop=True)
