import face_recognition
import cv2
import numpy as np
import psycopg2
import os

def capture_and_save_image(video_capture, save_directory, image_name):
    ret, frame = video_capture.read()
    if ret:
        image_path = os.path.join(save_directory, image_name)
        cv2.imwrite(image_path, frame)
        print(f"Image saved at: {image_path}")
        return frame, image_path
    else:
        print("Failed to capture image.")
        return None, None

def store_face_in_database(face_id, name, face_encoding, image_path):
    try:
        conn = psycopg2.connect(database="hrm_portal", user="postgres", password="arsha0612", host="localhost", port="5432")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO faces (id, name, encoding, image_path) VALUES (%s, %s, %s, %s)",
                       (face_id, name, face_encoding.tobytes(), image_path))
        conn.commit()
        conn.close()
        print(f"Stored face for ID {face_id} in database.")
    except Exception as e:
        print(f"Error storing face in database: {e}")

# Usage
save_directory = "C:/Users/God/Desktop/Web_Development/Face Recognition/Captured_Images"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not access the webcam.")
    exit()

face_id = input("Enter ID: ")
name = input("Enter name: ")

conditions = ['normal', 'angle1', 'angle2', 'without_specs', 'angle4']  # Conditions for capturing images
captured_images = 0

while True:
    for condition in conditions:
        print(f"\nCondition {captured_images % len(conditions) + 1}: {condition}")
        print("Press 'q' to capture the face or 'ESC' to exit.")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture frame.")
                continue

            # Display the frame
            cv2.imshow('Video Feed', frame)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # Capture image on 'q' key press
                image_name = f"{name}_{face_id}_{condition}.jpg"
                image_path = os.path.join(save_directory, image_name)
                cv2.imwrite(image_path, frame)
                print(f"Image saved at: {image_path}")

                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                if face_encodings:
                    store_face_in_database(face_id, name, face_encodings[0], image_path)
                    print(f"Stored face {captured_images + 1} for ID {face_id} under condition '{condition}' in database.")
                    captured_images += 1
                    break
                else:
                    print("Face not found in the captured image.")
            elif key == 27:  # Exit on 'ESC' key press
                print("Exiting...")
                break

        if key == 27:  # Exit loop if 'ESC' is pressed
            break

    # Prompt the user to continue or exit after capturing 5 images
    print("Press 'q' to continue capturing images or 'ESC' to exit.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Continue capturing
            break
        elif key == 27:  # Exit
            print("Exiting...")
            video_capture.release()
            cv2.destroyAllWindows()
            exit()

video_capture.release()
cv2.destroyAllWindows()
