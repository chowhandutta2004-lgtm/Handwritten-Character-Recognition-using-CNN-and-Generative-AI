import numpy as np
from tensorflow.keras.datasets import mnist
from generator_model import build_generator
from tensorflow.keras.models import load_model

def train_gan(latent_dim=100, epochs=5000, batch_size=128, save_path="../models/gan_model.h5"):
    """
    Trains a simple GAN generator to create handwritten digits (or letters).
    """
    # Load MNIST dataset (for GAN training)
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]

    # Build generator
    generator = build_generator(latent_dim=latent_dim)

    # Dummy training loop (for demonstration; full GAN needs discriminator and adversarial training)
    # Here we just save the generator so it can produce synthetic images
    generator.save(save_path)
    print(f"💾 GAN generator saved to {save_path}")

if __name__ == "__main__":
    train_gan()
