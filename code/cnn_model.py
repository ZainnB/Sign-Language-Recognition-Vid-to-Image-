import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# === Config ===
IMG_WIDTH = 128
IMG_HEIGHT = 128
NUM_CLASSES = 26
BATCH_SIZE = 4
EPOCHS = 20

LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# === Utility: Load images ===
def load_images(data_dir):
    X, y = [], []
    for img_name in sorted(os.listdir(data_dir)):
        if img_name.endswith('.png'):
            path = os.path.join(data_dir, img_name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0  # Normalize
            X.append(img)
            y.append(img_name[0])  # A, B, C... from filename
    X = np.array(X).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
    y = LabelEncoder().fit_transform(y)
    y = to_categorical(y, num_classes=NUM_CLASSES)
    return X, y

# === CNN Architecture ===
def build_model():
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        MaxPooling2D(),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(),
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === Train and Evaluate on One Landmark ===
def train_and_evaluate(landmark_name):
    print(f"\n Training on {landmark_name} images : ")

    train_dir = f"data/landmark_images/train/{landmark_name}"
    test_dir = f"data/landmark_images/train/{landmark_name}"

    X_train, y_train = load_images(train_dir)
    X_train
    X_test, y_test = load_images(test_dir)

    model = build_model()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    y_pred = model.predict(X_test)
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_true_labels, y_pred_labels)
    print(f"Accuracy of {landmark_name} is: {acc*100:.2f}%")

    return model, acc

# === Optional: Train and Evaluate on top 5 aggregated ===
# def train_and_evaluate():
#     print(f"\n Training on aggregated images : ")

#     train_dir = f"data/aggregated/train"
#     test_dir = f"data/aggregated/test"

#     X_train, y_train = load_images(train_dir)
#     X_test, y_test = load_images(test_dir)

#     model = build_model()
#     model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

#     y_pred = model.predict(X_test)
#     y_true_labels = np.argmax(y_test, axis=1)
#     y_pred_labels = np.argmax(y_pred, axis=1)

#     acc = accuracy_score(y_true_labels, y_pred_labels)
#     print(f"Accuracy: {acc*100:.2f}%")

#     return model, acc



# === Main Runner ===
if __name__ == "__main__":
    all_accuracies = {}

    for lm_name in LANDMARK_NAMES:
        _, acc = train_and_evaluate(lm_name)
        all_accuracies[lm_name] = acc

    # === Sort and Print Top 5 ===
    print("\nLandmark-wise Accuracy Summary:")
    sorted_acc = sorted(all_accuracies.items(), key=lambda x: x[1], reverse=True)
    for lm, acc in sorted_acc:
        print(f"{lm}: {acc*100:.2f}%")

    print("\nTop 5 Landmarks:")
    for i in range(5):
        lm, acc = sorted_acc[i]
        print(f"{i+1}. {lm} → {acc*100:.2f}%")

    # Save top-5 for next stage (feature aggregation)
    with open("top5_landmarks.txt", "w") as f:
        for lm, _ in sorted_acc[:5]:
            f.write(f"{lm}\n")

    print("\nSaved top-5 landmark names in top5_landmarks.txt")

# if __name__ == "__main__":
#     model, acc = train_and_evaluate()
#     print(f"\nFinal Model Accuracy: {acc*100:.2f}%")
#     # Save the trained model
#     model.save('my_model.keras')
#     print("Model saved as 'my_model.keras'")
