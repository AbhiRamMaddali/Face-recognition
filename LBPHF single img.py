# this program used to test single images 
import cv2
import numpy as np

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("fac_recognizer_model.yml")  # Load the trained model

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the test image
test_image_path = r"C:\Users\abhir\Desktop\OIP.jpg" # Replace with the path to your test image
test_image = cv2.imread(test_image_path)
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
imgre=cv2.resize(gray,(200,200))
# Detect faces in the test image
faces = face_cascade.detectMultiScale(imgre, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Recognize each detected face
for (x, y, w, h) in faces:
    face = imgre[y:y+h, x:x+w]  # Crop the face region
    label, confidence = recognizer.predict(face)  # Recognize the face

    # Draw a rectangle around the face and display the label
    cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    text = f"Person: {label}, Confidence: {confidence:.2f}"
    cv2.putText(test_image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
output_image = cv2.resize(test_image, (200, 200)) 

# Display the result
cv2.imshow("Test Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
