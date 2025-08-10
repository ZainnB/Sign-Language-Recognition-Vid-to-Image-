import os
import cv2
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from individual_cnn import load_images, build_model

# === Paths ===
MODELS_DIR = "models"
OUTPUT_DIR = "aggregation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Label Encoder ===
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# === Load Top 5 Landmarks ===
with open("top5_landmarks.txt") as f:
    TOP_LANDMARKS = [line.strip() for line in f.readlines()]

# === Utility: Save Confusion Matrix ===
def save_confusion_matrix(y_true, y_pred, classes, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {filename}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}_cm.png"))
    plt.close()

# === Utility: Save Classification Report ===
def save_classification_report(y_true, y_pred, classes, filename):
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    with open(os.path.join(OUTPUT_DIR, f"{filename}_report.txt"), "w") as f:
        f.write(report)

# === Create Aggregated Dataset ===
def create_feature_aggregation_dataset(mode='train'):
    base_dir = f"PSL_Dictionary/landmark_images/{mode}"
    output_dir = f"PSL_Dictionary/aggregated/{mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    sign_names = [f.split('.')[0] for f in os.listdir(os.path.join(base_dir, TOP_LANDMARKS[0]))]
    
    for sign in sign_names:
        combined_img = np.zeros((128, 128), dtype=np.uint8)
        for landmark in TOP_LANDMARKS:
            img_path = os.path.join(base_dir, landmark, f"{sign}.png")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                combined_img = cv2.bitwise_or(combined_img, img)
        cv2.imwrite(os.path.join(output_dir, f"{sign}.png"), combined_img)

# === Train Feature Aggregation CNN ===
def train_aggregated_model():
    X_train, y_train = load_images("PSL_Dictionary/aggregated/train", label_encoder=label_encoder, fit_encoder=False)
    X_test, y_test = load_images("PSL_Dictionary/aggregated/test", label_encoder=label_encoder, fit_encoder=False)
    
    model = build_model(len(label_encoder.classes_))
    model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1)
    
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_true_classes, y_pred_classes)
    print(f"Feature Aggregation Accuracy: {acc*100:.2f}%")
    
    save_confusion_matrix(y_true_classes, y_pred_classes, label_encoder.classes_, "feature_aggregation")
    save_classification_report(y_true_classes, y_pred_classes, label_encoder.classes_, "feature_aggregation")
    
    return model, acc

# === Cache test datasets for all landmarks ===
def cache_test_data():
    cache = {}
    for lm in TOP_LANDMARKS:
        X, y = load_images(f"PSL_Dictionary/landmark_images/test/{lm}", label_encoder=label_encoder, fit_encoder=False)
        cache[lm] = (X, y)
    return cache

# === Consolidated Majority Voting ===
def consolidated_majority_voting(test_cache):
    models = {lm: load_model(os.path.join(MODELS_DIR, f"{lm}_model.keras")) for lm in TOP_LANDMARKS}
    all_preds = [np.argmax(models[lm].predict(test_cache[lm][0], verbose=0), axis=1) for lm in TOP_LANDMARKS]
    
    final_preds = []
    for i in range(len(all_preds[0])):
        votes = [preds[i] for preds in all_preds]
        vote_counts = Counter(votes)
        max_count = max(vote_counts.values())
        candidates = [cls for cls, count in vote_counts.items() if count == max_count]
        
        if len(candidates) > 1:
            confidences = [sum(models[lm].predict(test_cache[lm][0][i:i+1])[0][c] for lm in TOP_LANDMARKS) for c in candidates]
            winner = candidates[np.argmax(confidences)]
        else:
            winner = candidates[0]
        final_preds.append(winner)
    
    y_true = np.argmax(test_cache[TOP_LANDMARKS[0]][1], axis=1)
    acc = accuracy_score(y_true, final_preds)
    print(f"Majority Voting Accuracy: {acc*100:.2f}%")
    
    save_confusion_matrix(y_true, final_preds, label_encoder.classes_, "majority_voting")
    save_classification_report(y_true, final_preds, label_encoder.classes_, "majority_voting")
    
    return acc

# === Ensemble Learning ===
def ensemble_learning(test_cache, top_n=5):
    models = [load_model(os.path.join(MODELS_DIR, f"{lm}_model.keras")) for lm in TOP_LANDMARKS[:top_n]]
    
    X_test, y_test = test_cache[TOP_LANDMARKS[0]]
    all_preds = []
    for i, lm in enumerate(TOP_LANDMARKS[:top_n]):
        X_lm, _ = test_cache[lm]
        all_preds.append(models[i].predict(X_lm, verbose=0))
    
    avg_preds = np.mean(all_preds, axis=0)
    y_pred_classes = np.argmax(avg_preds, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_true_classes, y_pred_classes)
    print(f"Ensemble Learning (Top {top_n}) Accuracy: {acc*100:.2f}%")
    
    if top_n == max(range(2, 6)):  # Save confusion/report only for last run
        save_confusion_matrix(y_true_classes, y_pred_classes, label_encoder.classes_, f"ensemble_{top_n}")
        save_classification_report(y_true_classes, y_pred_classes, label_encoder.classes_, f"ensemble_{top_n}")
    
    return acc

# === Save overall results ===
def save_results(method_name, accuracy):
    with open(os.path.join(OUTPUT_DIR, "aggregation_results.txt"), "a") as f:
        f.write(f"{method_name}: {accuracy:.4f}\n")

# === Plot ensemble results ===
def plot_ensemble_results(ensemble_results):
    plt.bar(ensemble_results.keys(), ensemble_results.values())
    plt.xlabel("Number of Landmarks")
    plt.ylabel("Accuracy")
    plt.title("Ensemble Performance")
    plt.savefig(os.path.join(OUTPUT_DIR, "ensemble_results.png"))

# === Main Execution ===
if __name__ == "__main__":
    print("=== Creating Aggregated Datasets ===")
    create_feature_aggregation_dataset('train')
    create_feature_aggregation_dataset('test')
    
    print("\n=== Feature Aggregation Training ===")
    _, agg_acc = train_aggregated_model()
    save_results("Feature_Aggregation", agg_acc)
    
    print("\n=== Caching Test Data ===")
    test_cache = cache_test_data()
    
    print("\n=== Consolidated Majority Voting ===")
    mv_acc = consolidated_majority_voting(test_cache)
    save_results("Majority_Voting", mv_acc)
    
    print("\n=== Ensemble Learning ===")
    ensemble_results = {}
    for n in range(2, 6):
        acc = ensemble_learning(test_cache, n)
        ensemble_results[n] = acc
        save_results(f"Ensemble_{n}_Landmarks", acc)
    
    with open(os.path.join(OUTPUT_DIR, "ensemble_results.json"), "w") as f:
        json.dump(ensemble_results, f, indent=4)
    
    plot_ensemble_results(ensemble_results)
    best_n = max(ensemble_results, key=ensemble_results.get)
    print(f"\nBest Ensemble Accuracy with Top {best_n} Landmarks: {ensemble_results[best_n]*100:.2f}%")
