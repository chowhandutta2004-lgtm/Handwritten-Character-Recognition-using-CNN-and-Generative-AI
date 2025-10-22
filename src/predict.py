import cv2
import numpy as np
from tensorflow.keras.models import load_model
import argparse

def predict_image(model_path, image_path):
    """
    Loads a trained model and predicts a handwritten character from image_path.
    """
    model = load_model(model_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0
    prediction = np.argmax(model.predict(img))
    print(f"🧩 Predicted Character: {prediction}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../dataset/test/sample.png",
                        help="Path to the input image")
    args = parser.parse_args()

    predict_image("../models/cnn_model.h5", args.image)
