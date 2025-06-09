import face_recognition
import cv2
import numpy as np
import joblib

def svmcamera():
    clf = joblib.load('encodings.joblib')
    names = joblib.load('name.joblib')
    centroid = joblib.load('averages.joblib')

    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)

            try:
                if face_locations and all(isinstance(loc, tuple) and len(loc) == 4 for loc in face_locations):
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                else:
                    face_encodings = []
            except Exception as e:
                print(f"[ERROR] face_encodings failed: {e}")
                face_encodings = []

            face_names = []
            if face_encodings:
                face_names = clf.predict(face_encodings)
                for (name, enc) in zip(face_names, face_encodings):
                    j = names.index(name)
                    if np.linalg.norm(centroid[j] - enc) > 0.40:
                        face_names[list(face_names).index(name)] = 'unknown'

        # Removed redundant line causing scope issue

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def person():
    clf = joblib.load('encodings.joblib')
    names = joblib.load('name.joblib')
    centroid = joblib.load('averages.joblib')

    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []
    process_this_frame = True

    max_attempts = 11  # Limit to ~30 seconds
    attempts = 0

    while attempts < max_attempts:
        ret, frame = video_capture.read()
        if not ret:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)

            if face_locations:
                try:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                except Exception as e:
                    print(f"[ERROR] face_encodings failed: {e}")
                    face_encodings = []

                if face_encodings:
                    predicted_names = clf.predict(face_encodings)
                    for name, enc in zip(predicted_names, face_encodings):
                        j = names.index(name)
                        dist = np.linalg.norm(centroid[j] - enc)
                        print(f"[DEBUG] Detected: {name}, Distance: {dist:.3f}")
                        if dist > 0.30:
                            video_capture.release()
                            cv2.destroyAllWindows()
                            return [names[j], j]

            else:
                print("[DEBUG] No faces detected.")

        process_this_frame = not process_this_frame
        attempts += 1

    # If we exit the loop
    video_capture.release()
    cv2.destroyAllWindows()
    print("[ERROR] Face not recognized within time limit.")
    return ["unknown", -1]


       


if __name__ == "__main__":
    svmcamera()