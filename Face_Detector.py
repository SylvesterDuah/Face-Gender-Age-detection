# Face_Detection.py



# import the computer vision 2
import cv2

# Random color for the rectangle
from random import randrange

# Load the model for gender detection
gender_model = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
gender_list = ['Male', 'Female']



# Load the model for age detection
age_model = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
age_list = ['0-2', '4-6', '8-12', '15-20', '21-24', '25-32', '33-43', '44-53', '60-100']


# Load some pre-trained data on face frontals from OpenCV (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Check if the webcam was successfully opened
if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# Iterate over frames
while True:
    # Read current frame
    successful_frame_read, frame = webcam.read()

    # Check if the frame was read successfully
    if not successful_frame_read:
        print("Failed to read frame from webcam")
        break

    # Convert it to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)

        # Crop the detected face
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict age
        age_model.setInput(blob)
        age_preds = age_model.forward()
        age = age_list[age_preds[0].argmax()]

        # Put gender and age text above rectangle
        cv2.putText(frame, f"{gender}, {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)



    # Display the video with detected faces
    cv2.imshow('Sly Detector', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam outside the loop
webcam.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Print message to indicate the script is complete
print("Mission Accomplished")



"""
# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

# Display the image with detected faces
cv2.imshow('Sylvester & Theresa', img)

# To hold the picture for long
cv2.waitKey()


# Doing this to know my code is working
print("Mission Accomplished")
"""



