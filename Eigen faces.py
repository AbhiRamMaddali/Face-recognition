#this program consist of training a dataset and program of Eigen faces program

import cv2
import numpy as np
import os

# Function to load the dataset
def load_dataset(path, target_size=(100, 100)):
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
                    img = cv2.resize(img, target_size)  # Resize to a consistent size
                    faces.append(img)
                    labels.append(current_label)
            current_label += 1

    return faces, labels, label_dict

# Load the dataset
dataset_path = r"C:\Users\abhir\Desktop\New folder (5)" # Path to your dataset
faces, labels, label_dict = load_dataset(dataset_path)

# Convert lists to NumPy arrays
faces = np.array(faces)
labels = np.array(labels)

# Create the Eigenfaces recognizer
recognizer = cv2.face.EigenFaceRecognizer_create()

# Train the model
recognizer.train(faces, labels)

# Save the trained model
recognizer.save("eigenface_model.yml")
print("Eigenfaces model trained and saved successfully.")
######################################## Load the trained Eigenfaces model
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read("eigenface_model.yml")

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Recognize each detected face
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Crop the face region
        face = cv2.resize(face, (100, 100))  # Resize to match training size
        label, confidence = recognizer.predict(face)  # Recognize the face

        # Set a confidence threshold
        if confidence < 5000:  # Adjust this threshold as needed
            text = f"Person: {label_dict[label]}, Confidence: {confidence:.2f}"
        else:
            text = "Unknown Person"

        # Draw a rectangle around the face and display the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with recognized faces
    cv2.imshow("Real-Time Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
