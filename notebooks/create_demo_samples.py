import numpy as np
import os

# Path to your full dataset
DATA_PATH = "../data/bodmas.npz"

# Folder where 1-sample files will be saved
OUTPUT_DIR = "../data/demo_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
data = np.load(DATA_PATH)
X, y = data["X"], data["y"]

print(f"Loaded dataset: X={X.shape}, y={y.shape}")

# Pick 10 random indices (without replacement)
rng = np.random.default_rng(42)
indices = rng.choice(len(X), size=10, replace=False)

for i, idx in enumerate(indices, start=1):
    sample = X[idx].reshape(1, -1)  # keep 2D shape (1, 2381)
    label = int(y[idx])

    # Save each as its own .npz
    out_path = os.path.join(OUTPUT_DIR, f"sample_{i}_label{label}.npz")
    np.savez_compressed(out_path, X=sample, y=np.array([label]))
    print(f"✅ Saved {out_path}")

print("\nAll demo samples created successfully!")
