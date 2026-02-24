import os
import requests

from configs import BOT_TOKEN, CHAT_ID

def send_telegram_alert(message, image_path=None):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data = {"chat_id": CHAT_ID, "text": message}

        )

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as img:
                requests.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                    data={"chat_id": CHAT_ID},
                    files={"photo": img}
                )

        
        print("Telegram alert sent")

    except Exception as e:
        print(f" Telegram alert failed: {e}")


# def send_whatsapp_alert(message):
#     """Send WhatsApp message asynchronously using pywhatkit"""
#     try:
#         def send_msg():
#             kit.sendwhatmsg_instantly(
#                 phone_no=WHATSAPP_NUMBER,
#                 message=message,
#                 wait_time=10,
#                 tab_close=True
#             )
#             print("✅ WhatsApp alert sent successfully")

#         # Run in a separate thread so main video loop doesn’t freeze
#         threading.Thread(target=send_msg, daemon=True).start()

#     except Exception as e:
#         print(f"❌ WhatsApp alert failed: {e}")
