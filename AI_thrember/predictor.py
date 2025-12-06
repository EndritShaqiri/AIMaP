import torch
import torch.nn as nn
import numpy as np
import joblib
import lightgbm as lgb
import json
import hashlib
from pathlib import Path
from .features import PEFeatureExtractor


# ============================================================
# Load paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

binary_model = lgb.Booster(model_file=str(MODEL_DIR / "aimap_binary_classifier.txt"))
scaler = joblib.load(MODEL_DIR / "family_scaler.pkl")

with open(MODEL_DIR / "family_label_to_idx.json", "r") as f:
    label_to_idx = {int(k): int(v) for k, v in json.load(f).items()}

with open(MODEL_DIR / "family_idx_to_label.json", "r") as f:
    idx_to_label = {int(k): int(v) for k, v in json.load(f).items()}

with open(MODEL_DIR / "recovered_family_map.json", "r") as f:
    family_names = json.load(f)

num_classes = len(idx_to_label)


# ============================================================
# Deep MLP (must match training exactly)
# ============================================================
class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Predictor class
# ============================================================
class AIMaPPredictor:
    def __init__(self):
        self.extractor = PEFeatureExtractor()
        self.input_dim = self.extractor.dim

        # Load family classifier
        self.family_model = DeepMLP(self.input_dim, num_classes)
        state = torch.load(MODEL_DIR / "family_mlp_best.pt", map_location="cpu")
        self.family_model.load_state_dict(state)
        self.family_model.eval()

    def extract(self, bytez: bytes):
        fv = self.extractor.feature_vector(bytez)
        return fv.reshape(1, -1)

    def predict(self, bytez: bytes):
        # ================================
        # Hashes
        # ================================
        md5_hash = hashlib.md5(bytez).hexdigest()
        sha1_hash = hashlib.sha1(bytez).hexdigest()
        sha256_hash = hashlib.sha256(bytez).hexdigest()

        # ================================
        # Raw PE meta
        # ================================
        raw = self.extractor.raw_features(bytez)

        # Import count
        imports_count = 0
        if "ImportsInfo" in raw and raw["ImportsInfo"]:
            imports_count = len(raw["ImportsInfo"].get("imports", []))

        # Section count
        sections_count = 0
        if "SectionInfo" in raw and raw["SectionInfo"]:
            sections_count = len(raw["SectionInfo"].get("sections", []))

        # ================================
        # Extract features
        # ================================
        fv = self.extract(bytez)

        # ======================================================
        # 1. Binary classifier (malicious probability)
        # ======================================================
        mal_prob = float(binary_model.predict(fv)[0])
        is_mal = mal_prob >= 0.5

        result = {
            "malicious_probability": mal_prob,
            "is_malicious": bool(is_mal),
            "family": None,
            "family_confidence": None,

            # Metadata
            "md5": md5_hash,
            "sha1": sha1_hash,
            "sha256": sha256_hash,
            "imports": imports_count,
            "sections": sections_count,
            "size": len(bytez)
        }

        # ======================================================
        # 2. Family classifier (only when malware = True)
        # ======================================================
        if is_mal:
            fv_norm = scaler.transform(fv)
            fv_tensor = torch.tensor(fv_norm, dtype=torch.float32)

            with torch.no_grad():
                logits = self.family_model(fv_tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]

            fam_idx = int(np.argmax(probs))
            conf = float(probs[fam_idx])
            fam_id = idx_to_label[fam_idx]

            fam_name = family_names.get(str(fam_id), f"Family_{fam_id}")

            result["family"] = fam_name
            result["family_confidence"] = conf

        return result
