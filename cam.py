# This program is used for  take images from webcam and stores images  in a folder
import cv2
import os

# Define the dataset path (use a valid path on your system)
dataset_path = r"c:\Users\abhir\Desktop\newdata"    #   Change this to a valid path

# Create the dataset directory (and any missing parent directories)
os.makedirs(dataset_path, exist_ok=True)  # Use exist_ok=True to avoid errors if the directory already exists

# Input the name of the person
person_name = input("Enter the name of the person: ")
person_path = os.path.join(dataset_path, person_name)

# Create the person's directory
os.makedirs(person_path, exist_ok=True)  # Use exist_ok=True to avoid errors if the directory already exists

# Initialize webcam
cap = cv2.VideoCapture(0)
count = 0

print("Press 's' to save an image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Display the frame
    cv2.imshow("Capture Images", frame)

    # Save the image when 's' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        img_name = os.path.join(person_path, f"{person_name}_{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        count += 1
    elif key == ord('q'):  # Quit when 'q' is pressed
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
