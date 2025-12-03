from .model import create_vectorized_features

if __name__ == "__main__":
    data_dir = r"C:\\Users\\Thinkbook 14\\AIMaP\\data\\ember_data"
    create_vectorized_features(data_dir)
    print("DONE.")
