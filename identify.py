# identify.py
import face_recognition
import os
import cv2
import numpy as np

def calculate_blurriness(image_crop):
    """Calculates a sharpness score for a cropped image. Higher is sharper."""
    if len(image_crop.shape) == 3:
        gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_crop
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def setup_known_faces(known_faces_dir='known_faces'):
    """Scans the known_faces directory and learns each face."""
    known_face_encodings = []
    known_face_names = []
    print("[INFO] Loading known faces for identification...")
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(person_name)
                except Exception as e:
                    print(f"[WARNING] Could not load or process image {image_path}: {e}")
    print(f"[SUCCESS] {len(known_face_names)} known faces loaded.")
    return known_face_encodings, known_face_names

def identify_and_crop_faces(image_array, known_face_encodings, known_face_names):
    """
    Finds all faces, sorts them into known and unknown, and returns their
    names and cropped images (in RGB format).
    """
    known_person_data = []
    unknown_face_crops = []

    face_locations = face_recognition.face_locations(image_array)
    face_encodings = face_recognition.face_encodings(image_array, face_locations)

    for i, face_encoding in enumerate(face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.52)
        top, right, bottom, left = face_locations[i]
        padding = 20
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(image_array.shape[0], bottom + padding)
        right = min(image_array.shape[1], right + padding)
        face_crop = image_array[top:bottom, left:right]

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            known_person_data.append({'name': name, 'crop': face_crop})
        else:
            unknown_face_crops.append(face_crop)
            
    return known_person_data, unknown_face_crops