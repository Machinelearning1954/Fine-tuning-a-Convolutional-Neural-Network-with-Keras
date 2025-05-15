# This script combines preprocessing, model definition, compilation, and training.
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

print("Starting the complete fine-tuning process...")

# 1. Load and preprocess the CIFAR-10 dataset
print("Step 1: Loading and preprocessing CIFAR-10 dataset...")
(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
print("- CIFAR-10 dataset loaded.")

# Normalize the pixel values to [0, 1]
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print("- Pixel values normalized.")

# One-hot encode the labels
y_train_full = to_categorical(y_train_full, 10)
y_test = to_categorical(y_test, 10)
print("- Labels one-hot encoded.")

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42)
print("- Data split into training (90%) and validation (10%) sets.")
print(f"  x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"  x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
print(f"  x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
print("Step 1 finished.")

# 2. Choose pre-trained CNN model (VGG16) and customize top layers
print("\nStep 2: Choosing and customizing VGG16 model...")
input_shape = (32, 32, 3) # CIFAR-10 images are 32x32x3

base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
print("- VGG16 base model loaded (ImageNet weights, top excluded).")

for layer in base_model.layers:
    layer.trainable = False
print("- Base model layers frozen.")

x = base_model.output
x = GlobalAveragePooling2D()(x)
print("- GlobalAveragePooling2D layer added.")
x = Dense(256, activation="relu")(x)
print("- Dense layer (256 units, ReLU) added.")
predictions = Dense(10, activation="softmax")(x)
print("- Classification layer (10 units, softmax) added.")

model = Model(inputs=base_model.input, outputs=predictions)
print("- Fine-tuned model created.")
print("Step 2 finished.")

# 3. Compile and configure transfer learning pipeline
print("\nStep 3: Compiling the model...")
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
print("- Model compiled with Adam optimizer (lr=0.001), categorical_crossentropy loss, and accuracy metric.")
model.summary()
print("Step 3 finished.")

# 4. Train fine-tuned model on dataset
print("\nStep 4: Training the fine-tuned model...")
epochs = 10
batch_size = 32 # A common batch size, can be tuned

# Optional: Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print(f"- Starting training for {epochs} epochs with batch_size={batch_size}...")
history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    callbacks=[early_stopping] # Add early stopping here
)

print("- Model training finished.")
print("Step 4 finished.")

# Save the trained model
model.save("/home/ubuntu/cifar10_vgg16_finetuned.h5")
print("\nTrained model saved to /home/ubuntu/cifar10_vgg16_finetuned.h5")

# Print history for quick check (in a notebook, this would be plotted)
print("\nTraining History:")
print(history.history)

print("\nAll steps for training completed.")

