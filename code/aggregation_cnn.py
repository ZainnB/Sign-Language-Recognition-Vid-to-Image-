import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from individual_cnn import load_images, build_model
from collections import Counter

global label_encoder
label_encoder = LabelEncoder()

# Load top 5 landmarks from previous step
with open("top5_landmarks.txt") as f:   
    TOP_LANDMARKS = [line.strip() for line in f.readlines()]

def create_feature_aggregation_dataset(mode='train'):
    """Create dataset where each sample combines top 5 landmarks"""
    base_dir = f"PSL_Dictionary/landmark_images/{mode}"
    output_dir = f"PSL_Dictionary/aggregated/{mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all sign names (assuming same across landmarks)
    sign_names = [f.split('.')[0] for f in os.listdir(os.path.join(base_dir, TOP_LANDMARKS[0]))]
    
    for sign in sign_names:
        combined_img = np.zeros((128, 128), dtype=np.uint8)
        
        for landmark in TOP_LANDMARKS:
            img_path = os.path.join(base_dir, landmark, f"{sign}.png")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            combined_img = cv2.bitwise_or(combined_img, img)  # Combine landmarks
        
        cv2.imwrite(os.path.join(output_dir, f"{sign}.png"), combined_img)

# Create datasets
create_feature_aggregation_dataset('train')
create_feature_aggregation_dataset('test')

# Train CNN on aggregated features
def train_aggregated_model():

    X_train, y_train = load_images("PSL_Dictionary/aggregated/train", fit_encoder=True)
    X_test, y_test = load_images("PSL_Dictionary/aggregated/test")
    
    model = build_model(len(label_encoder.classes_))
    model.fit(X_train, y_train, epochs=50, batch_size=4)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print(f"Feature Aggregation Accuracy: {acc*100:.2f}%")
    
    return model


def consolidated_majority_voting():
    """Combine predictions from top 5 models using majority voting"""
    # Load all models
    models = {lm: load_model(f"models/{lm}_model.keras") for lm in TOP_LANDMARKS}
    
    # Load test data for each landmark
    test_data = {}
    for lm in TOP_LANDMARKS:
        X, _ = load_images(f"PSL_Dictionary/landmark_images/test/{lm}")
        test_data[lm] = X
    
    # Get predictions from each model
    all_preds = []
    for lm in TOP_LANDMARKS:
        preds = models[lm].predict(test_data[lm], verbose=0)
        all_preds.append(np.argmax(preds, axis=1))
    
    # Majority voting with tie-breaking
    final_preds = []
    for i in range(len(all_preds[0])):
        votes = [pred[i] for pred in all_preds]
        
        # Count votes
        vote_counts = Counter(votes)
        max_count = max(vote_counts.values())
        candidates = [k for k,v in vote_counts.items() if v == max_count]
        
        # Tie-breaking (use confidence scores)
        if len(candidates) > 1:
            confidences = []
            for c in candidates:
                conf = sum([models[lm].predict(test_data[lm][i:i+1])[0][c] 
                           for lm in TOP_LANDMARKS])
                confidences.append(conf)
            winner = candidates[np.argmax(confidences)]
        else:
            winner = candidates[0]
            
        final_preds.append(winner)
    
    # Calculate accuracy
    _, y_test = load_images(f"PSL_Dictionary/landmark_images/test/{TOP_LANDMARKS[0]}")
    y_true = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true, final_preds)
    print(f"Consolidated Majority Voting Accuracy: {acc*100:.2f}%")

def ensemble_learning(top_n=5):
    """Train ensemble of top N models and combine predictions"""
    # Load top N models
    models = []
    for lm in TOP_LANDMARKS[:top_n]:
        model = load_model(f"models/{lm}_model.keras")
        models.append((lm, model))
    
    # Evaluate ensemble
    all_preds = []
    for lm, model in models:
        X_test, y_test = load_images(f"PSL_Dictionary/landmark_images/test/{lm}")
        preds = model.predict(X_test, verbose=0)
        all_preds.append(preds)
    
    # Weighted average (can also use majority voting)
    avg_preds = np.mean(all_preds, axis=0)
    y_pred = np.argmax(avg_preds, axis=1)
    
    # Get true labels (same for all landmarks)
    _, y_test = load_images(f"PSL_Dictionary/landmark_images/test/{TOP_LANDMARKS[0]}")
    y_true = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Ensemble Learning (Top {top_n}) Accuracy: {acc*100:.2f}%")
    
    return acc

def save_results(method_name, accuracy):
    with open("aggregation_results.txt", "a") as f:
        f.write(f"{method_name}: {accuracy:.4f}\n")

def plot_ensemble_results(ensemble_results):
    plt.bar(ensemble_results.keys(), ensemble_results.values())
    plt.xlabel("Number of Landmarks")
    plt.ylabel("Accuracy")
    plt.title("Ensemble Performance")
    plt.savefig("ensemble_results.png")

if __name__ == "__main__":
    print("=== Running Feature Aggregation ===")
    aggregated_model = train_aggregated_model()
    
    print("\n=== Running Consolidated Majority Voting ===")
    consolidated_majority_voting()
    
    print("\n=== Running Ensemble Learning ===")
    ensemble_results = {}
    for n in range(2, 6):  # Test top 2-5 landmarks
        acc = ensemble_learning(n)
        ensemble_results[n] = acc
    save_results(f"Ensemble_{n}_Landmarks", acc)
    plot_ensemble_results(ensemble_results)
    best_n = max(ensemble_results, key=ensemble_results.get)
    print(f"Best Ensemble Accuracy with Top {best_n} Landmarks: {ensemble_results[best_n]*100:.2f}%")

    # Final comparison
    print("\n=== Final Results ===")
    print("1. Feature Aggregation: Trained single model on combined landmarks")
    print("2. Consolidated Dataset: Majority voting across top models")
    print("3. Ensemble Learning: Averaged predictions from multiple models")
    
    # Save best ensemble
    best_n = max(ensemble_results, key=ensemble_results.get)
    print(f"\nBest performing ensemble uses top {best_n} landmarks")