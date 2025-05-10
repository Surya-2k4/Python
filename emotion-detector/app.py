import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained emotion detection model
model = load_model('emotion_detection_model.h5')

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load emotion labels
with open('labels.txt', 'r') as f:
    emotion_labels = [line.strip() for line in f.readlines()]

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop and preprocess face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

        # Predict emotion
        predictions = model.predict(roi_reshaped)
        max_index = int(np.argmax(predictions))
        emotion = emotion_labels[max_index]

        # Display label above face
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
