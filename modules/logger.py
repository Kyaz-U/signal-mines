import logging

logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)

def log_event(message):  # Bu yangi funksiya — xatolikni tuzatadi
    logging.info(f"[EVENT] {message}")
