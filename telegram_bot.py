import os
import telebot
import pandas as pd
from dotenv import load_dotenv
from predict_mines import predict_safest_cells
from modules.logger import log_event
from modules.csv_checker import check_csv_integrity
from modules.model_guard import validate_input_shape

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

CSV_PATH = "data/mines_data.csv"
MODEL_PATH = "models/mines_rf_models.pkl"

def init_csv():
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w") as f:
            header = ",".join([f"cell_{i+1}" for i in range(25)]) + ",bombs_count\n"
            f.write(header)

@bot.message_handler(commands=['start'])
def start(message):
    init_csv()
    bot.send_message(message.chat.id, "AI Mines Signal Bot ishga tushdi. /bombs orqali ma'lumot kiriting.")

@bot.message_handler(commands=['bombs'])
def save_bombs(message):
    parts = message.text.split()[1:]
    bombs = set(int(p) for p in parts if p.isdigit() and 1 <= int(p) <= 25)
    if len(bombs) != 3:
        bot.send_message(message.chat.id, "Iltimos, aynan 3 ta bombali katak kiriting (masalan: /bombs 5 12 21)")
        return

    row = [1 if i+1 in bombs else 0 for i in range(25)]
    row.append(len(bombs))
    with open(CSV_PATH, "a") as f:
        f.write(",".join(map(str, row)) + "\n")

    bot.send_message(message.chat.id, "Yozib olindi. Model yangilanmoqda...")
    os.system("python train_model.py")
    bot.send_message(message.chat.id, "AI modeli yangilandi!")
    log_event(f"[+] Bombs: {bombs} modelga qo‘shildi.")

@bot.message_handler(commands=['signal'])
def send_signal(message):
    try:
        df = pd.read_csv(CSV_PATH)
        is_valid_csv, check_msg = check_csv_integrity(df)
        if not is_valid_csv:
            bot.send_message(message.chat.id, f"CSV muammo: {check_msg}")
            log_event(check_msg)
            return

        if len(df) < 5:
            bot.send_message(message.chat.id, "Kamida 5 ta o‘yin natijasi kerak.")
            log_event("[X] Yetarlicha data mavjud emas")
            return

        avg_row = df.tail(5).mean().drop("bombs_count").values.reshape(1, -1)
        is_valid_input, shape_msg = validate_input_shape(avg_row)
        if not is_valid_input:
            bot.send_message(message.chat.id, shape_msg)
            log_event(f"[X] {shape_msg}")
            return

        safest = predict_safest_cells(df)
        text = "Eng xavfsiz kataklar: " + ", ".join(safest)
        bot.send_message(message.chat.id, text)
        with open("chart.png", "rb") as photo:
            bot.send_photo(message.chat.id, photo)
        log_event(f"[+] Signal yuborildi: {text}")

    except Exception as e:
        bot.send_message(message.chat.id, f"Xatolik: {str(e)}")
        log_event(f"[X] Signal xatolik: {str(e)}")

if __name__ == "__main__":
    init_csv()
    bot.polling()
