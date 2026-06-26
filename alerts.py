# alerts.py (FINAL FIXED VERSION)

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import cv2
from twilio.rest import Client
from twilio.http.http_client import TwilioHttpClient
from PIL import Image
import io

# --- Your Email Credentials ---
EMAIL_SENDER = 'system@gmail.com'
EMAIL_PASSWORD = '16letters password'
EMAIL_RECEIVER = 'user@gmail.com'

# --- Your Twilio Credentials ---
TWILIO_SID = 'XXXXXXXXXXXXXXXXXXXXXXXXXX'
TWILIO_AUTH_TOKEN = 'XXXXXXXXXXXXXXXXX'
TWILIO_PHONE_NUMBER = 'XXXXXXX'
RECEIVER_PHONE_NUMBER = 'User number'
TWIML_BIN_URL = 'https://handler.twilio.com/.......'

# --- Timeout ---
NETWORK_TIMEOUT = 15


# ✅ FIXED IMAGE ATTACH FUNCTION
def attach_image_from_array(msg, image_array, filename):
    try:
        img = image_array.copy()

        # Convert to uint8 if needed
        if img.dtype != 'uint8':
            img = (img * 255).astype('uint8')

        # Ensure correct color format
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL
        pil_img = Image.fromarray(img)

        # Save to buffer
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        buffer.seek(0)

        msg.attach(MIMEImage(buffer.read(), name=filename))

    except Exception as e:
        print(f"[EMAIL ERROR] Failed to attach {filename}: {e}")


# --- SMS ---
def send_sms_alert(body):
    try:
        http_client = TwilioHttpClient(timeout=NETWORK_TIMEOUT)
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN, http_client=http_client)
        client.messages.create(
            body=body,
            from_=TWILIO_PHONE_NUMBER,
            to=RECEIVER_PHONE_NUMBER
        )
        print("[SUCCESS] SMS sent.")
    except Exception as e:
        raise RuntimeError(f"SMS failed: {e}")


# --- CALL ---
def make_phone_call_alert():
    try:
        http_client = TwilioHttpClient(timeout=NETWORK_TIMEOUT)
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN, http_client=http_client)
        call = client.calls.create(
            url=TWIML_BIN_URL,
            to=RECEIVER_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER
        )
        print("[SUCCESS] Call initiated.")
    except Exception as e:
        raise RuntimeError(f"Call failed: {e}")


# --- EMAIL ---
def send_email_with_all_photos(subject, body, event_frame=None, known_person_data=None, unknown_face_crops=None):
    try:
        print("[DEBUG] Preparing email...")

        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # ✅ Attach main frame
        if event_frame is not None:
            attach_image_from_array(msg, event_frame, "event_snapshot.jpg")

        # ✅ Attach known faces
        if known_person_data:
            for person in known_person_data:
                crop = person.get("crop")
                name = person.get("name", "known_person")
                if crop is not None:
                    attach_image_from_array(msg, crop, f"{name}.jpg")

        # ✅ Attach unknown faces
        if unknown_face_crops:
            for i, face_crop in enumerate(unknown_face_crops):
                if face_crop is not None:
                    attach_image_from_array(msg, face_crop, f"unknown_{i+1}.jpg")

        print("[DEBUG] Sending email...")

        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=NETWORK_TIMEOUT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()

        print("[SUCCESS] Email sent with images.")

    except Exception as e:
        raise RuntimeError(f"Email failed: {e}")
