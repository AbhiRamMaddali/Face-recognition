# this program is used make datset file i.e yml
import cv2
import numpy as np
import os

# Function to load the preprocessed dataset
def load_dataset(path):
    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            label_dict[current_label] = dir_name
            subject_path = os.path.join(root, dir_name)
            for file_name in os.listdir(subject_path):
                if file_name.endswith(".jpg"):
                    img_path = os.path.join(subject_path, file_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    faces.append(img)
                    labels.append(current_label)
            current_label += 1

    return faces, labels, label_dict

# Load the dataset
dataset_path = r"C:\Users\abhir\Desktop\New folder (5)"
faces, labels, label_dict = load_dataset(dataset_path)

# Train the LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Save the trained model
recognizer.save("fac_recognizer_model.yml")
print("Model trained and saved successfully.")
