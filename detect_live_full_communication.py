# detect_live_full_communication.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque
import pygame
import datetime
import threading
import os
import time

# Import our custom modules
from alerts import send_email_with_all_photos, send_sms_alert, make_phone_call_alert
from identify import setup_known_faces, identify_and_crop_faces, calculate_blurriness
from webcam_stream import WebcamStream

# --- CONFIGURATION & PARAMETERS ---
MODEL_PATH = 'violence_detection_model_advanced.h5'
ALARM_SOUND_PATH = 'sounds/alarm.wav'
UNKNOWN_FACES_DIR = 'unknown_faces'
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.85
ALERT_COOLDOWN_SECONDS = 3 # 5 minutes
EVIDENCE_BUFFER_LENGTH = 30
PREDICTION_QUEUE_LENGTH = 15
MIN_FACE_SIZE = 60
MIN_BLUR_SCORE = 50.0

if not os.path.exists(UNKNOWN_FACES_DIR):
    os.makedirs(UNKNOWN_FACES_DIR)

# --- Global variable for cooldown ---
last_alert_time = 0

# --- BACKGROUND THREAD FUNCTION ---
def process_full_alert(frame_buffer, known_encodings, known_names):
    try:
        print("[THREAD] Full alert thread started.")
        best_faces = {}
        best_overall_frame = None
        highest_frame_blur_score = -1
        known_person_data=[]
        unknown_face_crops=[]

        for frame in frame_buffer:
                if frame is None:
                    continue
                if frame.dtype !='uint8':
                    frame=(frame*255).astype('uint8')
                    if len(frame.shape)==2:
                        frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        try:
                            known_person_data, unknown_face_crops = identify_and_crop_faces(rgb_frame, known_encodings, known_names)
                        except Exception as e:
                            print(f"[Warning] Face detection failed: {e}")
                            continue

                for person in known_person_data:
                    try:
                        name= person.get('name','unknown')
                        crop= person.get('crop',None)
                        if crop is None: 
                            continue
                        blur_score = calculate_blurriness(crop)
                        if crop.shape[0] < MIN_FACE_SIZE or blur_score < MIN_BLUR_SCORE: continue
                        if name not in best_faces or blur_score > best_faces[name]['quality']:
                         best_faces[name] = {'crop': crop, 'quality': blur_score}
                    except Exception as e:
                        print(f"[Warning] Error processing known face: {e}")
                        continue

                for i, crop in enumerate(unknown_face_crops):
                 try:
                     if crop is None:
                         continue
                     blur_score = calculate_blurriness(crop)
                     if crop.shape[0] < MIN_FACE_SIZE or blur_score < MIN_BLUR_SCORE: continue
                     unknown_id = f'unknown_{i}'
                     if unknown_id not in best_faces or blur_score > best_faces[unknown_id]['quality']:
                      best_faces[unknown_id] = {'crop': crop, 'quality': blur_score}
                 except Exception as e:
                    print(f"[Warning] Error processing unknown face: {e}")
                    continue
        
        final_known_data, final_unknown_crops = [], []
        for key, data in best_faces.items():
            if "unknown" in key: final_unknown_crops.append(data['crop'])
            else: final_known_data.append({'name': key, 'crop': data['crop']})

        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = "URGENT: Violence Detected!"
        sms_body = f"ALERT: Violence detected at {timestamp_str}. Check email for details."
        email_body = f"Violence detected at {timestamp_str}."

        # DISPATCH ALL ALERTS
        make_phone_call_alert()
        send_sms_alert(sms_body)
        send_email_with_all_photos(subject, email_body, best_overall_frame, final_known_data, final_unknown_crops)
        
        print("[THREAD] All alerts dispatched successfully.")
    except Exception as e:
        # If any alert fails, this will print the error from the background thread
        print(f"[THREAD ERROR] An exception occurred in the alert thread: {e}")

# --- MAIN DETECTION FUNCTION ---
def run_live_detection_final():
    global last_alert_time
    pygame.mixer.init()
    try: pygame.mixer.music.load(ALARM_SOUND_PATH)
    except: print("[WARNING] Alarm sound not found.")
    model = load_model(MODEL_PATH)
    known_encodings, known_names = setup_known_faces()
    predictions_queue = deque(maxlen=PREDICTION_QUEUE_LENGTH)
    evidence_buffer = deque(maxlen=EVIDENCE_BUFFER_LENGTH)
    alarm_sounding = False
    vs = WebcamStream(src=0).start()
    time.sleep(2.0)

    print("[INFO] Running FINAL detection with full communication... Press 'q' to quit.")

    while True:
        frame = vs.read()
        if frame is None: continue
        evidence_buffer.append(frame.copy())
        
        frame_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame_array = img_to_array(frame_resized) / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)
        prediction = model.predict(frame_array, verbose=0)[0][0]
        predictions_queue.append(prediction)
        
        # CORRECTED: Calculate avg_prediction correctly
        avg_prediction = np.mean(predictions_queue) if predictions_queue else 0
        
        if avg_prediction > CONFIDENCE_THRESHOLD:
            label = f"VIOLENCE ({avg_prediction * 100:.1f}%)"
            color = (0, 0, 255)
            if not alarm_sounding:
                pygame.mixer.music.play(-1)
                alarm_sounding = True
                current_time = time.time()
                if (current_time - last_alert_time) > ALERT_COOLDOWN_SECONDS:
                    last_alert_time = current_time
                    print("[INFO] Cooldown passed. Launching full communication alert thread.")
                    alert_thread = threading.Thread(
                        target=process_full_alert, 
                        args=(list(evidence_buffer), known_encodings, known_names)
                    )
                    alert_thread.start()
        else:
            label = f"Non-Violence ({(1 - avg_prediction) * 100:.1f}%)"
            color = (0, 255, 0)
            if alarm_sounding:
                pygame.mixer.music.stop()
                alarm_sounding = False
        
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow('Final Violence Detection System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    print("[INFO] Stopping system...")
    if alarm_sounding: pygame.mixer.music.stop()
    vs.stop()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == '__main__':
    run_live_detection_final()