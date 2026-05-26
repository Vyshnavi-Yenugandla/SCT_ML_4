import cv2
import os

gesture_name = input("Enter gesture name: ")

dataset_path = "dataset/" + gesture_name

os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        f"Images Collected: {count}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collecting Data", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):

        img_name = f"{dataset_path}/{count}.jpg"

        cv2.imwrite(img_name, frame)

        count += 1

        print("Saved:", img_name)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()