import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

data = []
labels = []

dataset_path = "dataset"

for gesture_name in os.listdir(dataset_path):

    gesture_folder = os.path.join(dataset_path, gesture_name)

    for image_name in os.listdir(gesture_folder):

        image_path = os.path.join(gesture_folder, image_name)

        img = cv2.imread(image_path)

        img = cv2.resize(img, (64, 64))

        img = img.flatten()

        data.append(img)

        labels.append(gesture_name)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2
)

model = SVC()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

joblib.dump(model, "gesture_model.pkl")

print("Model Saved")