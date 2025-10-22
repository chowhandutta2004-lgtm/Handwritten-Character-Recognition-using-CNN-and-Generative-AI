"""
# 🧠 SRC Folder Documentation — Handwritten Character Recognition using CNN & Generative AI

This folder contains the core source code for the project **Handwritten Character Recognition using CNN and Generative AI**.  
Below is a description of each script and its purpose.

---

## 📘 File Descriptions

### 1️⃣ cnn_model.py  
Defines the **Convolutional Neural Network (CNN)** architecture that extracts features from handwritten images and classifies them into respective characters or digits.

**Key Layers:**
- Conv2D + MaxPooling2D layers for feature extraction  
- Dense layers for classification  
- Dropout for regularization  

---

### 2️⃣ generator_model.py  
Implements a simple **Generative Adversarial Network (GAN) Generator** to create synthetic handwritten-like samples.  
This improves dataset diversity and helps the CNN generalize better.

**Key Features:**
- Dense + LeakyReLU layers for non-linear learning  
- Reshape layer to output 28x28 synthetic images  

---

### 3️⃣ train_model.py  
Trains the CNN model using the MNIST dataset (digits) or your own dataset.  
You can replace MNIST with EMNIST for alphabets.

**Workflow:**
1. Load and preprocess data  
2. Train the CNN model  
3. Evaluate accuracy  
4. Save the trained model to `/models/cnn_model.h5`  

---

### 4️⃣ predict.py  
Loads the trained CNN model and predicts handwritten characters from an input image.

**Usage Example:**
```bash
python predict.py --image ../dataset/test/sample.png

Install all required libraries:

pip install tensorflow keras numpy opencv-python pandas matplotlib seaborn scikit-learn

1️⃣ Train the model:

python train_model.py


2️⃣ Predict a new image:

python predict.py

cnn_model.py
=====================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_cnn_model(input_shape=(28, 28, 1), num_classes=36):
"""
Builds a CNN model for handwritten character recognition.
"""
model = Sequential([
Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
MaxPooling2D(2,2),
Conv2D(64, (3,3), activation='relu'),
MaxPooling2D(2,2),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.4),
Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
return model

=====================================================
generator_model.py
=====================================================

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

=====================================================
train_model.py
=====================================================

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


if name == "main":
train_cnn()

=====================================================
predict.py
=====================================================

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

if name == "main":
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default="../dataset/test/sample.png", help="Path to the input image")
args = parser.parse_args()

predict_image("../models/cnn_model.h5", args.image)
