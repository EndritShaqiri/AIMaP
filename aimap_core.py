import os
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
import lightgbm as lgb
from pathlib import Path
from AI_thrember.features import PEFeatureExtractor


# ======================================================================
# Paths
# ======================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# Family MLP
# ======================================================================
class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ======================================================================
# Load models and preprocessing artifacts
# ======================================================================
binary_model = lgb.Booster(model_file=str(MODEL_DIR / "aimap_binary_classifier.txt"))
family_scaler = joblib.load(MODEL_DIR / "family_scaler.pkl")

with open(MODEL_DIR / "family_label_to_idx.json", "r") as f:
    label_to_idx = {int(k): int(v) for k, v in json.load(f).items()}

with open(MODEL_DIR / "family_idx_to_label.json", "r") as f:
    idx_to_label = {int(k): int(v) for k, v in json.load(f).items()}

num_classes = len(idx_to_label)
extractor = PEFeatureExtractor()

_family_model = None  # lazy load


def _load_family_model(input_dim):
    global _family_model
    if _family_model is None:
        model = DeepMLP(input_dim, num_classes)
        state = torch.load(MODEL_DIR / "family_mlp_best.pt", map_location=_device)
        model.load_state_dict(state)
        model.to(_device)
        model.eval()
        _family_model = model
    return _family_model


# ======================================================================
# Main prediction function
# ======================================================================
def predict_file(file_bytes):
    """file_bytes = uploaded file content"""

    # 1. Extract features
    feature_vec = extractor.feature_vector(file_bytes)
    feature_vec_2d = feature_vec.reshape(1, -1)

    # 2. Predict malicious probability (binary classifier)
    mal_prob = float(binary_model.predict(feature_vec_2d)[0])
    is_malicious = mal_prob >= 0.5  # adjustable threshold

    result = {
        "malicious_probability": mal_prob,
        "is_malicious": bool(is_malicious),
        "family": None,
        "family_confidence": None,
    }

    # 3. Family classifier only runs if malicious
    if is_malicious:
        fv_norm = family_scaler.transform(feature_vec_2d)
        fv_t = torch.from_numpy(fv_norm).float().to(_device)

        model = _load_family_model(feature_vec.shape[0])
        with torch.no_grad():
            logits = model(fv_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        fam_idx = int(np.argmax(probs))
        conf = float(probs[fam_idx])
        fam_id = idx_to_label[fam_idx]

        result["family"] = fam_id
        result["family_confidence"] = conf

    return result
