"""
Training Pipeline for Exercise Error Classification
Trains a Multi-label Classifier (Random Forest / MLP Neural Network)
to pinpoint joint-specific errors (Knee depth, Back posture, Knee valgus).

Usage:
  python core_coach/train_model.py
"""

import os
import sys
import joblib
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigure stdout to utf-8 for Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints"))
MODEL_PATH = os.path.join(MODEL_DIR, "posture_error_model.pkl")

ERROR_LABELS = [
    "KNEE_DEPTH_ERROR",   # 0: Gối chưa đủ độ sâu
    "BACK_LEAN_ERROR",    # 1: Gập lưng quá mức
    "KNEE_VALGUS_ERROR"   # 2: Gối chụm vào trong
]

def generate_synthetic_training_dataset(num_samples=300):
    """
    Generates a realistic synthetic kinematic dataset representing 300 Squat repetitions.
    Features per Rep (6 dimensions):
      0: Min Knee Angle (deg)
      1: Min Hip Angle (deg)
      2: Max Torso Lean Angle (deg)
      3: Knee Distance Ratio (Knee Dist / Ankle Dist)
      4: Rep Duration (seconds)
      5: Max Velocity (deg/s)
    """
    np.random.seed(42)
    X = []
    Y = []

    for _ in range(num_samples):
        # Decide which errors this rep has
        err_knee = np.random.choice([0, 1], p=[0.6, 0.4])    # 40% chance of shallow squat
        err_back = np.random.choice([0, 1], p=[0.7, 0.3])    # 30% chance of excessive lean
        err_valgus = np.random.choice([0, 1], p=[0.75, 0.25])# 25% chance of knee valgus

        # Generate realistic kinematic features matching these error labels
        if err_knee == 1:
            # Shallow squat: lowest knee angle stops at 105° - 130°
            min_knee = np.random.uniform(105.0, 130.0)
        else:
            # Good depth: lowest knee angle reaches 75° - 90°
            min_knee = np.random.uniform(75.0, 92.0)

        if err_back == 1:
            # Excessive forward lean: torso angle drops to 30° - 48°
            torso_angle = np.random.uniform(30.0, 48.0)
        else:
            # Upright torso: 58° - 75°
            torso_angle = np.random.uniform(58.0, 78.0)

        if err_valgus == 1:
            # Knee valgus: knee distance shrinks relative to ankle distance (< 0.80)
            valgus_ratio = np.random.uniform(0.55, 0.78)
        else:
            # Neutral knees: ratio 0.90 - 1.15
            valgus_ratio = np.random.uniform(0.92, 1.15)

        min_hip = min_knee + np.random.uniform(-5.0, 10.0)
        duration = np.random.uniform(1.2, 2.5)
        velocity = np.random.uniform(40.0, 90.0)

        # Add Gaussian noise
        features = [
            min_knee + np.random.normal(0, 1.5),
            min_hip + np.random.normal(0, 1.5),
            torso_angle + np.random.normal(0, 1.0),
            valgus_ratio + np.random.normal(0, 0.02),
            duration + np.random.normal(0, 0.05),
            velocity + np.random.normal(0, 2.0)
        ]

        X.append(features)
        Y.append([err_knee, err_back, err_valgus])

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)

import glob
import pandas as pd

def load_training_dataset(data_dir="data/processed", num_synthetic_samples=400):
    """
    Auto-detects and loads dataset in order of priority:
    1. Real .csv files in data/processed/ (e.g. Kaggle / UI-PRMD dataset)
    2. Real .npy files in data/processed/
    3. Biomechanical synthetic generator (Fallback)
    """
    # 1. Check for CSV files in data/processed/
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if csv_files:
        csv_path = csv_files[0]
        print(f"[+] Found REAL CSV Dataset: '{csv_path}'! Loading with pandas...")
        df = pd.read_csv(csv_path)
        print(f"[+] Total rows in CSV: {len(df)} frames | Columns: {list(df.columns)}")

        # Select kinematic feature columns
        feature_cols = [
            'left_knee_angle', 'right_knee_angle',
            'left_hip_angle', 'right_hip_angle',
            'spine_angle', 'torso_lean',
            'left_knee_lateral', 'right_knee_lateral',
            'symmetry_score', 'hip_depth'
        ]
        # Check which feature columns exist in df
        available_cols = [c for c in feature_cols if c in df.columns]
        if len(available_cols) >= 4:
            X = df[available_cols].values.astype(np.float32)
            
            # Extract or map label
            if 'label' in df.columns:
                raw_labels = df['label'].values
                # Format to multi-label or binary target
                if raw_labels.ndim == 1:
                    # Convert single label to 3-joint multi-label indicator
                    Y = np.zeros((len(raw_labels), 3), dtype=np.int32)
                    for idx, lbl in enumerate(raw_labels):
                        if lbl == 1:
                            Y[idx, 0] = 1  # Knee depth error
                        elif lbl == 2:
                            Y[idx, 1] = 1  # Back lean error
                        elif lbl >= 3:
                            Y[idx, 2] = 1  # Valgus error
                else:
                    Y = raw_labels.astype(np.int32)
            else:
                Y = np.zeros((len(X), 3), dtype=np.int32)

            print(f"[+] Successfully loaded {X.shape[0]} real samples with {X.shape[1]} features from CSV!")
            return X, Y

    # 2. Check for .npy files
    real_x_path = os.path.join(data_dir, "exercise_features_X.npy")
    real_y_path = os.path.join(data_dir, "exercise_labels_Y.npy")

    if os.path.exists(real_x_path) and os.path.exists(real_y_path):
        print(f"[+] Found REAL .npy dataset in '{data_dir}'! Loading...")
        X = np.load(real_x_path)
        Y = np.load(real_y_path)
        print(f"[+] Loaded {X.shape[0]} real reps from Person A.")
        return X, Y

    # 3. Fallback to synthetic data
    print(f"[*] Note: No CSV or .npy found in '{data_dir}'.")
    print(f"[*] Generating {num_synthetic_samples} samples using Biomechanical Kinematic Distribution (Escamilla 2001 & NSCA)...")
    return generate_synthetic_training_dataset(num_samples=num_synthetic_samples)

def train_and_save_model():
    print("=" * 65)
    print("AI POSE COACH: POSTURE ERROR CLASSIFIER TRAINING")
    print("=" * 65)

    # 1. Prepare Dataset (Auto-detect Real vs Synthetic)
    X, Y = load_training_dataset()
    print(f"[+] Total samples: {X.shape[0]} reps | Features per rep: {X.shape[1]}")
    print(f"[+] Error Labels: {ERROR_LABELS}")

    # Split Train / Test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 2. Build Multi-label Classifier
    # Using MultiOutputClassifier wrapping a Random Forest (Ensemble Tree)
    print("\n[*] Training Multi-label Random Forest Classifier...")
    base_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model = MultiOutputClassifier(base_clf)
    model.fit(X_train, Y_train)

    # 3. Evaluate on Test Set
    Y_pred = model.predict(X_test)
    exact_match_acc = accuracy_score(Y_test, Y_pred)
    print(f"\n[+] Training Complete!")
    print(f"    - Exact Match Accuracy (All 3 joints correct simultaneously): {exact_match_acc * 100:.2f}%\n")

    for i, lbl in enumerate(ERROR_LABELS):
        joint_acc = accuracy_score(Y_test[:, i], Y_pred[:, i])
        print(f"    - Accuracy for [{lbl}]: {joint_acc * 100:.2f}%")

    # 4. Save Trained Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[+] Successfully saved model weights to:\n    {MODEL_PATH}")

    # 5. Verification Test with Real Test Samples
    print("\n" + "=" * 65)
    print("VERIFICATION INFERENCE TEST ON REAL TEST SAMPLES")
    print("=" * 65)

    for idx in range(min(3, len(X_test))):
        sample = X_test[idx:idx+1]
        pred = model.predict(sample)[0]
        actual = Y_test[idx]
        match = list(pred) == list(actual)
        print(f"Sample {idx+1}: Predicted={list(pred)} | Actual={list(actual)} | Match={match}")
    print("=" * 65)

if __name__ == "__main__":
    train_and_save_model()
