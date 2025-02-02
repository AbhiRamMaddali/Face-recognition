# this propram is used for preprocessing like image resizeing and converting RGBimage into gray scale and normalization ike brightness.

import cv2
import os

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to detect and crop faces
def preprocess_images(input_path, output_path, target_size=(100, 100)):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for person_name in os.listdir(input_path):
        person_input_path = os.path.join(input_path, person_name)
        person_output_path = os.path.join(output_path, person_name)
        if not os.path.exists(person_output_path):
            os.makedirs(person_output_path)

        for img_name in os.listdir(person_input_path):
            img_path = os.path.join(person_input_path, img_name)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, target_size)
                output_img_path = os.path.join(person_output_path, img_name)
                cv2.imwrite(output_img_path, face_resized)

# Preprocess the dataset
input_dataset_path = r"c:\Users\abhir\Desktop\newdata"   
output_dataset_path = r"C:\Users\abhir\Desktop\New folder (4)"
preprocess_images(input_dataset_path, output_dataset_path)
