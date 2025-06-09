import face_recognition
import sklearn as sc
from sklearn import svm
import os
import joblib
import numpy
import itertools
import openpyxl

def svm_weight_create():
    encodings = []
    names = []
    total = 0
    avg = []
    avg_dist = 0
    store_dist = []
    face_enc123_person = [] # Renamed to avoid confusion within the loop
    train_dir = os.listdir('images/')

    # Loop through each person in the training directory
    for person in train_dir:
        person_image_folder = os.path.join("images", person)
        if not os.path.isdir(person_image_folder):
            print(f"Skipping {person_image_folder} as it's not a directory.")
            continue

        pix = os.listdir(person_image_folder)
        i = 0
        person_total_encoding = numpy.zeros(128) # Initialize as a zero vector
        face_enc123_person = [] # Reset for each person

        # Loop through each training image for the current person
        for person_img in pix:
            image_path = os.path.join(person_image_folder, person_img)
            print(f"Processing image: {image_path}") # Debug print
            try:
                # Get the face encodings for the face in each image file
                face = face_recognition.load_image_file(image_path)
                # This returns a LIST of encodings (or an empty list)
                face_enc_list = face_recognition.face_encodings(face)

                # *** Check if any face was found ***
                if face_enc_list:
                    # Use the first face found in the image
                    face_enc = face_enc_list[0]

                    # Add face encoding for current image with corresponding label (name)
                    encodings.append(face_enc)
                    face_enc123_person.append(face_enc) # Add to this person's list
                    names.append(person)
                    person_total_encoding = numpy.add(person_total_encoding, face_enc)
                    i = i + 1
                else:
                    # *** Handle case where no face is found in the image ***
                    print(f"Warning: No face detected in {image_path}. Skipping this image.")

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue # Skip to the next image if loading/processing fails

        # --- Calculations for the current person (moved outside the inner loop) ---
        if i > 0:
            # Calculate average encoding for this person
            person_avg = person_total_encoding / i
            avg.append(person_avg)

            # Calculate average distance for this person if more than one image was processed
            if i > 1:
                res = itertools.combinations(face_enc123_person, 2)
                d = 0
                current_person_avg_dist = 0
                for each in res:
                    d = d + 1
                    current_person_avg_dist += numpy.linalg.norm(each[1] - each[0])

                if d > 0: # Should always be true if i > 1
                    current_person_avg_dist /= d
                    store_dist.append(current_person_avg_dist)
                    print(f"Avg distance for {person}: {current_person_avg_dist}")
                else: # Should not happen if i > 1, but as a safeguard
                    print(f"Warning: Could not calculate distance for {person} despite {i} images.")
                    store_dist.append(0.0)
            else: # Only 1 image processed for this person
                print(f"Warning: Only 1 image processed for {person}. Cannot calculate average distance.")
                store_dist.append(0.0) # Append a placeholder

        else:
            # Handle case where no images were successfully processed for the person
            print(f"Warning: No images successfully processed for {person} in {person_image_folder}. Skipping average calculation.")
            # Do not append to avg or store_dist if no images were processed

        # No need to reset totals here, they are initialized at the start of the outer loop

    # --- Training and Saving (outside the main loop) ---
    if not encodings:
        print("Error: No face encodings were generated. Cannot train the classifier.")
        return # Exit if no data

    print(f"Training classifier with {len(encodings)} encodings and {len(names)} names.")
    # Create and train the SVC classifier
    clf = svm.SVC(gamma='scale', kernel='poly', probability=True) # Enable probability if needed later
    clf.fit(encodings, names)

    # --- Logic to store unique encodings/names (potentially flawed, review needed) ---
    # This part seems to intend to store only one encoding per person,
    # but it relies on the order which might not be guaranteed or desired.
    # It might be better to store all encodings or rethink this logic based on need.
    # For now, keeping the original logic but adding checks.

    store_encodings_unique = []
    name_store_unique = []
    processed_names = set()

    # Iterate through all collected encodings and names
    for encoding, name in zip(encodings, names):
        if name not in processed_names:
            store_encodings_unique.append(encoding)
            name_store_unique.append(name)
            processed_names.add(name)

    # Check if any unique names were found before saving
    if not name_store_unique:
        print("Warning: No unique names found to save in name.joblib/pics.joblib.")
    else:
        print(f"Saving {len(name_store_unique)} unique names/encodings.")
        joblib.dump(store_encodings_unique, "pics.joblib")
        joblib.dump(name_store_unique, "name.joblib")

    # Save averages, classifier, and distances
    joblib.dump(avg, "averages.joblib")
    joblib.dump(clf, "encodings.joblib")
    joblib.dump(store_dist, "distance_avg.joblib") # Corrected typo

    print("SVM weight creation complete.")

if __name__ == "__main__":
    svm_weight_create()