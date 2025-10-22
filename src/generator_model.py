from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape
import numpy as np

def build_generator(latent_dim=100, output_shape=(28, 28, 1)):
    """
    Builds a simple generator network for synthetic data augmentation.
    """
    model = Sequential([
        Dense(256, input_dim=latent_dim),
        LeakyReLU(0.2),
        Dense(512),
        LeakyReLU(0.2),
        Dense(1024),
        LeakyReLU(0.2),
        Dense(np.prod(output_shape), activation='tanh'),
        Reshape(output_shape)
    ])
    return model
