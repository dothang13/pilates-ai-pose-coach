"""
Machine Learning / Deep Learning Error Classifier
Predicts joint-specific error probabilities (Multi-label classification: Knee, Spine/Back, Hip).
Supports scikit-learn (Random Forest / MLP Neural Net), PyTorch, and ONNX Runtime.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple

DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints", "posture_error_model.pkl")
)

ERROR_LABELS = [
    "KNEE_DEPTH_ERROR",   # Error 0: Chưa đủ độ sâu khớp gối
    "BACK_LEAN_ERROR",    # Error 1: Gập lưng / Võng thắt lưng quá mức
    "KNEE_VALGUS_ERROR"   # Error 2: Đầu gối chụm vào trong
]

ERROR_MESSAGES = {
    "KNEE_DEPTH_ERROR": "AI phát hiện: Khớp gối chưa đủ độ sâu!",
    "BACK_LEAN_ERROR": "AI phát hiện: Lưng đang bị cong / gập quá mức!",
    "KNEE_VALGUS_ERROR": "AI phát hiện: Đầu gối đang bị chụm vào trong!"
}

class PostureErrorClassifier:
    """
    Evaluator that loads trained Machine Learning weights to detect joint-level errors.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                print(f"[+] Loaded trained Error Classifier from: {self.model_path}")
            except Exception as e:
                print(f"[!] Warning: Could not load model: {e}")
        else:
            print(f"[*] Note: Model weights not found at '{self.model_path}'. Run 'python core_coach/train_model.py' to train.")

    def predict_errors(self, rep_feature_vector: np.ndarray) -> Tuple[bool, List[str], Dict[str, float]]:
        """
        Input: Feature vector of a repetition (Shape: [num_features] or [1, num_features])
        Returns: (has_error, error_messages, probabilities_dict)
        """
        if not self.is_loaded or self.model is None:
            # Fallback if model is not trained yet
            return False, [], {}

        features = np.array(rep_feature_vector).reshape(1, -1)
        
        # Predict binary multi-labels: shape [1, 3]
        predictions = self.model.predict(features)[0]

        # Predict probabilities if supported
        probs = {}
        if hasattr(self.model, "predict_proba"):
            try:
                # For MultiOutputClassifier, predict_proba is a list of arrays
                raw_probs = self.model.predict_proba(features)
                for i, lbl in enumerate(ERROR_LABELS):
                    # Probability of class 1 (error)
                    probs[lbl] = float(raw_probs[i][0][1]) if len(raw_probs[i][0]) > 1 else 0.0
            except Exception:
                for i, lbl in enumerate(ERROR_LABELS):
                    probs[lbl] = float(predictions[i])
        else:
            for i, lbl in enumerate(ERROR_LABELS):
                probs[lbl] = float(predictions[i])

        detected_errors = []
        for i, lbl in enumerate(ERROR_LABELS):
            if predictions[i] == 1 or probs.get(lbl, 0.0) >= 0.5:
                detected_errors.append(ERROR_MESSAGES[lbl])

        has_error = len(detected_errors) > 0
        return has_error, detected_errors, probs
