import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from cnn_model import build_cnn_model

def train_cnn():
    """
    Trains the CNN model on the MNIST dataset.
    Replace MNIST with EMNIST or your own dataset for alphabets.
    """
    # Load MNIST dataset (digits 0–9)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and reshape
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Convert labels to categorical (10 classes for digits 0–9)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build and train model
    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=10)
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    # Evaluate and save
    loss, acc = model.evaluate(x_test, y_test)
    print(f"✅ Test Accuracy: {acc*100:.2f}%")

    model.save("../models/cnn_model.h5")
    print("💾 Model saved to models/cnn_model.h5")

if __name__ == "__main__":
    train_cnn()
