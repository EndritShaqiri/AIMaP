# train_top20_model.py
import os
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

TOP20_INDICES = [637,2359,2360,2355,658,655,683,613,691,2364,
                 1546,95,2354,626,32,1695,930,255,2375,578]

def main():
    os.makedirs("models", exist_ok=True)
    data = np.load("../data/bodmas.npz")
    X = data["X"]
    y = data["y"]
    print("Loaded X shape:", X.shape)
    X_top20 = X[:, TOP20_INDICES]  # N x 20
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_top20, y, test_size=0.2, random_state=42, stratify=y
    )
    # Model
    model = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=64, random_state=42
    )
    print("Training top-20 LightGBM...")
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    try:
        print("AUC:", roc_auc_score(y_test, y_proba))
    except:
        pass
    print("Classification report:\n", classification_report(y_test, y_pred))
    joblib.dump(model, "models/lightgbm_top20.pkl")
    print("Saved models/lightgbm_top20.pkl")

if __name__ == "__main__":
    main()
