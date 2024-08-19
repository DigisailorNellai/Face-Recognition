import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from config import DATABASE
import face_recognition
import cv2
import psycopg2
import numpy as np
from datetime import datetime


def fetch_all_face_encodings():
    conn = psycopg2.connect(
        dbname=DATABASE['name'],
        user=DATABASE['user'],
        password=DATABASE['password'],
        host=DATABASE['host'],
        port=DATABASE['port']
    )
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, encoding FROM faces")
    rows = cursor.fetchall()

    conn.close()

    face_data = []
    for row in rows:
        face_id = row[0]
        name = row[1]
        encoding = np.frombuffer(row[2], dtype=np.float64)
        face_data.append((face_id, name, encoding))

    return face_data

def recognize_face():
    video_capture = cv2.VideoCapture(0)

    known_faces = fetch_all_face_encodings()
    print(f"Loaded {len(known_faces)} known faces.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image.")
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if not face_encodings:
            print("No faces detected.")
        else:
            print(f"Detected {len(face_encodings)} faces.")

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([face[2] for face in known_faces], face_encoding)
            face_distances = face_recognition.face_distance([face[2] for face in known_faces], face_encoding)

            if matches:
                best_match_index = np.argmin(face_distances)
                face_id = known_faces[best_match_index][0]
                name = known_faces[best_match_index][1]
                recognition_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Recognized: ID = {face_id}, Name = {name}, Time = {recognition_time}")
            else:
                print("Face not recognized.")

        cv2.imshow('Video', frame)

        # Add a delay for testing purposes
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_face()
