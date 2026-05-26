import cv2
import joblib
import numpy as np

model = joblib.load("gesture_model.pkl")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opening")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    roi = cv2.resize(frame, (64, 64))

    roi_flat = roi.flatten().reshape(1, -1)

    prediction = model.predict(roi_flat)[0]

    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Prediction", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()