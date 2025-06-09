import face_recognition
import os
import numpy as np
import joblib
from sklearn.svm import SVC
import cv2
import dlib

def train_face_recognition_from_images(dataset_path='images/', output_encodings='encodings.joblib', output_names='name.joblib', output_averages='averages.joblib', output_classifier='classifier.joblib'):
    """red
    Trains a face recognition model by generating face encodings from a dataset of images
    captured by create_face.py or similar, and trains an SVM classifier.

    Args:
        dataset_path (str): Path to the directory containing subdirectories of images for each person.
        output_encodings (str): Filename for saving the raw face encodings (optional).
        output_names (str): Filename for saving the corresponding names.
        output_averages (str): Filename for saving the average encoding for each person.
        output_classifier (str): Filename for saving the trained SVM classifier.
    """
    known_face_encodings = []
    known_face_names = []
    average_encodings = {}

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            encodings_for_person = []
            for filename in os.listdir(person_folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_folder, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_locations = face_recognition.face_locations(image)
                        if face_locations:
                            face_encoding = face_recognition.face_encodings(image, face_locations)[0]  # Take the first face
                            known_face_encodings.append(face_encoding)
                            known_face_names.append(person_name)
                            encodings_for_person.append(face_encoding)
                        else:
                            print(f"No face found in {image_path}")
                    except Exception as e:
                        print(f"Error loading or processing image {image_path}: {e}")

            if encodings_for_person:
                average_encoding = np.mean(encodings_for_person, axis=0)
                average_encodings[person_name] = average_encoding
            else:
                print(f"No encodings generated for {person_name}")

    if known_face_encodings:
        # Train the SVM classifier
        print("Training SVM classifier...")
        clf = SVC(kernel='linear', probability=True)
        clf.fit(known_face_encodings, known_face_names)
        print("SVM classifier trained.")

        # Save the trained classifier and other data
        joblib.dump(clf, output_classifier)
        joblib.dump(known_face_names, output_names)
        joblib.dump(average_encodings, output_averages)
        joblib.dump(known_face_encodings, output_encodings) # Optional: Save raw encodings

        print(f"Classifier saved to {output_classifier}")
        print(f"Names saved to {output_names}")
        print(f"Average encodings saved to {output_averages}")
        print(f"Encodings saved to {output_encodings} (optional)")
    else:
        print("No face encodings were generated. Please ensure there are detectable faces in the images.")

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def capture_faces_for_encoding(name="unknown", num_images=10, output_dir="images"):
    """
    Captures faces from the webcam, aligns them, and saves them into a specified directory
    for later encoding. This version performs basic alignment using eye detection.
    """
    detector = dlib.get_frontal_face_detector()
    try:
        shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    except Exception as e:
        print(f"Error loading shape predictor: {e}. Make sure it's in the same directory.")
        return

    face_dir = os.path.join(output_dir, name)
    create_folder(face_dir)
    img_no = 1
    cap = cv2.VideoCapture(0)
    print(f"Capturing {num_images} faces for {name}...")

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray)

        for face in faces:
            try:
                shape = shape_predictor(img_gray, face)
                left_eye = shape.part(36)
                right_eye = shape.part(45)
                dY = right_eye.y - left_eye.y
                dX = right_eye.x - left_eye.x
                angle = np.degrees(np.arctan2(dY, dX))

                M = cv2.getRotationMatrix2D((left_eye.x, left_eye.y), angle, 1.0)
                rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

                face_img = rotated[face.top():face.bottom(), face.left():face.right()]
                resized_face = cv2.resize(face_img, (200, 200)) # Standardize size

                img_path = os.path.join(face_dir, f"{name}_{img_no}.jpg")
                cv2.imwrite(img_path, resized_face)
                print(f"Saved: {img_path}")
                img_no += 1

                cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing face: {e}")

        cv2.imshow("Capturing Faces", img)
        if cv2.waitKey(1) & 0xFF == ord('q') or img_no > num_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Face capture complete.")

if __name__ == "__main__":
    create_folder("images")
    person_name = input("Enter the name for the person you want to encode: ")
    capture_faces_for_encoding(name=person_name, num_images=20) # Capture more images for better encoding
    print(f"\nNow encoding faces for {person_name}...")
    train_face_recognition_from_images(dataset_path="images", output_classifier="encodings.joblib", output_names="name.joblib", output_averages="averages.joblib")
    print("Encoding process finished. 'encodings.joblib', 'name.joblib', and 'averages.joblib' have been created/updated.")