import pywhatkit as kit
import threading
import time
import os

def send_whatsapp_alert_nonblocking(phone_no, message, image_path=None):
    """Send WhatsApp message + image in background using pywhatkit"""
    
    def _send():
        try:
            print("📲 Initiating WhatsApp alert...")

            # Step 1: Send the text message instantly
            time.sleep(2)  # ensure WhatsApp Web loads before sending
            kit.sendwhatmsg_instantly(
                phone_no, 
                message, 
                wait_time=20,  # give enough load time
                tab_close=True
            )
            print("✅ WhatsApp text message sent successfully.")
            
            # Step 2: Wait before sending the image
            if image_path and os.path.exists(image_path):
                print("🖼 Sending intrusion snapshot...")
                # wait to ensure previous browser tab closed properly
                time.sleep(15)
                kit.sendwhats_image(
                    receiver=phone_no,
                    img_path=image_path,
                    caption="📸 Intrusion Snapshot",
                    wait_time=20,
                    tab_close=True
                )
                print("✅ WhatsApp image sent successfully.")

        except Exception as e:
            print(f"❌ WhatsApp send failed: {e}")

    # Run in background so detection doesn’t freeze
    thread = threading.Thread(target=_send, daemon=True)
    thread.start()
