import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# === Config ===
IMG_WIDTH = 128
IMG_HEIGHT = 128
BATCH_SIZE = 4
EPOCHS = 50  # Increased from 20 to match paper's training duration
VALIDATION_SPLIT = 0.2  # For validation during training

# Landmark names matching MediaPipe indices
LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# === Utility: Load and preprocess images ===
def load_images(data_dir, label_encoder=None, fit_encoder=False):
    X, y = [], []
    for img_name in sorted(os.listdir(data_dir)):
        if img_name.endswith('.png'):
            path = os.path.join(data_dir, img_name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            # Verify binary image (0 or 255 values)
            assert set(np.unique(img)).issubset({0, 255}), f"Image {path} is not binary!"
            
            img = img / 255.0  # Normalize to [0,1]
            X.append(img)
            
            # Extract base sign name (removes _1/_2 suffixes)
            label = os.path.splitext(img_name)[0].rsplit('_', 1)[0]
            y.append(label)

    X = np.array(X).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
    # Encode labels
    if fit_encoder:
        y_encoded = label_encoder.fit_transform(y)
        print("\nClass distribution:", Counter(label_encoder.classes_))
    else:
        y_encoded = label_encoder.transform(y)
    
    y_categorical = to_categorical(y_encoded, num_classes=len(label_encoder.classes_))
    return X, y_categorical

# === CNN Architecture (Matches Paper Exactly) ===
def build_model(num_classes):
    model = Sequential([
        # Layer 1 (Paper Table 9)
        Conv2D(16, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        MaxPooling2D(),
        
        # Layer 2
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(),
        
        # Layer 3
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        
        # Layer 4
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(),
        
        # Layer 5
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(),
        
        # Classification head
        Flatten(),
        Dense(512, activation='sigmoid'),  # Paper uses sigmoid (Table 9)
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(),  # Paper Section 4.3.1
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    return model

# === Train and Evaluate One Landmark ===
def train_and_evaluate(landmark_name):
    print(f"\n{'='*50}\nTraining on {landmark_name} images\n{'='*50}")
    
    # Paths
    train_dir = f"PSL_Dictionary/landmark_images/train/{landmark_name}"
    test_dir = f"PSL_Dictionary/landmark_images/test/{landmark_name}"
    
    # Load data
    global label_encoder
    label_encoder = LabelEncoder()
    X_train, y_train = load_images(train_dir, label_encoder=label_encoder,fit_encoder=True)
    X_test, y_test = load_images(test_dir, label_encoder=label_encoder)
    
    # Build and train model
    model = build_model(num_classes=len(label_encoder.classes_))
    
    print("\nTraining Progress:")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )
    
    # Evaluate
    y_pred = model.predict(X_test, verbose=0)
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    acc = accuracy_score(y_true_labels, y_pred_labels)
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))
    
    return model, acc

# === Main Execution ===
if __name__ == "__main__":
    # Initialize
    label_encoder = LabelEncoder()
    all_accuracies = {}
    all_models = {}
    
    # Train models for all landmarks
    for lm_name in LANDMARK_NAMES:
        try:
            model, acc = train_and_evaluate(lm_name)
            all_accuracies[lm_name] = acc
            all_models[lm_name] = model
        except Exception as e:
            print(f"⚠️ Error processing {lm_name}: {str(e)}")
            continue
    
    # Save results
    os.makedirs("models", exist_ok=True)
    
    # 1. Save all models (for ensemble learning)
    for lm_name, model in all_models.items():
        model.save(f"models/{lm_name}_model.keras")
    
    # 2. Save top 5 models (for feature aggregation)
    sorted_acc = sorted(all_accuracies.items(), key=lambda x: x[1], reverse=True)
    top5 = sorted_acc[:5]
    
    os.makedirs("top5_models", exist_ok=True)
    with open("top5_landmarks.txt", "w") as f:
        for i, (lm_name, acc) in enumerate(top5, 1):
            print(f"{i}. {lm_name}: {acc*100:.2f}%")
            f.write(f"{lm_name}\n")
            all_models[lm_name].save(f"top5_models/{lm_name}_model.keras")
    
    print("\nSaved all models to 'models/' and top 5 to 'top5_models/'")