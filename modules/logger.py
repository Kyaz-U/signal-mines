import logging
import os

# Loglar uchun papka yaratish
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log fayl yo'li
LOG_FILE = os.path.join(LOG_DIR, "bot.log")

# Log sozlamalari
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_info(message):
    """Informatsion log yozish"""
    logging.info(message)

def log_error(message):
    """Xatolik logini yozish"""
    logging.error(message)

def log_event(message):
    """Maxsus hodisa logini yozish"""
    logging.info(f"[EVENT] {message}")
