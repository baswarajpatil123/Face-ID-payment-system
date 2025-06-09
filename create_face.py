import cv2
import numpy as np
import os, time
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def create_face():
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)

    FACE_DIR = "images/"
    create_folder(FACE_DIR)
    while True:
        name=input("EnterName: ")
        #face_id = input("Enter id for face: ")
        
        try:
            #face_id = int(face_id)
            face_folder = FACE_DIR + str(name) + "/"
            create_folder(face_folder)
            break
        except:
            print("Invalid input. id must be int")
            continue

    # get beginning image number
    while True:
        init_img_no = input("Starting img no.: ")
        try:
            init_img_no = int(init_img_no)
            break
        except:
            print("Starting img no should be integer...")
            continue

    img_no = init_img_no
    cap = cv2.VideoCapture(0)
    

    total_imgs = 10
    while True:
        ret, img = cap.read()
      
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.squeeze()
        
        faces = detector(img_gray)
       

       
       
        if len(faces) == 1:
            face = faces[0]
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_img = img_gray[y-50:y + h+100, x-50:x + w+100]
            face_aligned = face_aligner.align(img, img_gray, face)

            face_img = face_aligned
            img_path = face_folder +name+ str(img_no) + ".jpg"
            cv2.imwrite(img_path, face_img)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.imshow("aligned", face_img)
            img_no += 1

        cv2.imshow("Saving", img)
        cv2.waitKey(1)
        if img_no == init_img_no + total_imgs:
            break

    cap.release()

def create_face_website(name):
    detector = dlib.get_frontal_face_detector()
    # Ensure the path to this file is correct
    try:
        shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    except Exception as e:
        print(f"Error loading shape predictor: {e}")
        print("Ensure 'shape_predictor_68_face_landmarks.dat' is in the correct directory.")
        return # Stop if predictor can't be loaded
        
    # face_aligner is removed
    # face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)

    FACE_DIR = "images/"
    create_folder(FACE_DIR)
    # --- (Keep the folder creation logic) ---
    try:
        face_folder = FACE_DIR + str(name) + "/"
        create_folder(face_folder)
    except Exception as e:
        print(f"Error creating directory {face_folder}: {e}")
        return

    # --- (Keep the init_img_no logic) ---
    init_img_no = 1 # Simplified from previous version
    img_no = init_img_no

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return
        
    total_imgs = 10
    print("Starting face capture for registration...") # Add feedback

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            time.sleep(0.1) # Avoid busy-looping on error
            continue
            
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray)

        if len(faces) == 1:
            face = faces[0]
            # (x, y, w, h) = face_utils.rect_to_bb(face) # We don't need the simple crop now

            # --- Start Manual Alignment ---
            try:
                shape = shape_predictor(img_gray, face)
                shape_np = face_utils.shape_to_np(shape)

                # Extract left and right eye coordinates (indices 36-41 and 42-47)
                leftEyePts = shape_np[36:42]
                rightEyePts = shape_np[42:48]

                # Compute the center of mass for each eye
                leftEyeCenter = leftEyePts.mean(axis=0) # Keep as float for accuracy
                rightEyeCenter = rightEyePts.mean(axis=0)

                # Compute the angle between the eye centers
                dY = rightEyeCenter[1] - leftEyeCenter[1]
                dX = rightEyeCenter[0] - leftEyeCenter[0]
                angle = np.degrees(np.arctan2(dY, dX))

                # Compute the center coordinates between the eyes
                eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2.0,
                              (leftEyeCenter[1] + rightEyeCenter[1]) / 2.0)
                
                # Define desired face size (adjust as needed)
                desiredFaceWidth = 200
                output_shape = (desiredFaceWidth, desiredFaceWidth) 

                # Get the rotation matrix
                # Use the calculated float coordinates for eyesCenter
                M = cv2.getRotationMatrix2D(tuple(eyesCenter), angle, 1.0) 

                # Optional: Adjust translation matrix (helps center face in output)
                # Adjust translation to center the midpoint of the eyes
                tX = desiredFaceWidth * 0.5
                tY = desiredFaceWidth * 0.35 # Position eyes slightly above center
                M[0, 2] += (tX - eyesCenter[0])
                M[1, 2] += (tY - eyesCenter[1])


                # Apply the affine transformation
                face_aligned = cv2.warpAffine(img, M, output_shape, flags=cv2.INTER_CUBIC)
                
                # --- End Manual Alignment ---

                # Save the aligned face
                img_path = face_folder + name + str(img_no) + ".jpg"
                cv2.imwrite(img_path, face_aligned)
                print(f"Saved: {img_path}") # Add feedback

                # Optional: Display aligned face during capture (can slow things down)
                # cv2.imshow("Aligned Face", face_aligned) 
                
                img_no += 1

            except Exception as e_align:
                print(f"Error during face alignment or saving: {e_align}")
                # Continue to next frame if alignment fails for one frame

        # Optional: Display original feed (can slow things down)
        # cv2.imshow("Capturing...", img)
        
        # Use waitKey to allow windows to update if showing images
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'): # Allow manual quit during capture
        #     break

        if img_no > total_imgs: # Check > instead of == for safety
            print("Finished face capture.")
            break

    cap.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    create_face()
