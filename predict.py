import sys
from pathlib import Path
from aimap_core import predict_file   # your inference engine

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <file>")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    
    if not filepath.exists():
        print(f"Error: File '{filepath}' does not exist.")
        sys.exit(1)

    with open(filepath, "rb") as f:
        file_bytes = f.read()

    result = predict_file(file_bytes)
    
    print("\n=== AIMaP Prediction ===")
    print(f"Malicious Probability: {result['malicious_probability']:.4f}")
    print(f"Is Malicious:         {result['is_malicious']}")

    if result["is_malicious"]:
        print(f"Predicted Family:     {result['family']}")
        print(f"Confidence:           {result['family_confidence']:.4f}")
    print("========================\n")

if __name__ == "__main__":
    main()
