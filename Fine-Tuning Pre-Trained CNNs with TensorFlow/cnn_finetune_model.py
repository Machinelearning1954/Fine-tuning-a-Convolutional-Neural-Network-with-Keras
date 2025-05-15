import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load the CIFAR-10 dataset input shape (needed for VGG16 input_shape)
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# input_shape = x_train.shape[1:] # Should be (32, 32, 3)
input_shape = (32, 32, 3) # CIFAR-10 images are 32x32x3

# Load the pre-trained VGG16 model (excluding the top classifier)
print("Loading VGG16 base model...")
base_model = VGG16(weights=\'imagenet\', include_top=False, input_shape=input_shape)
print("VGG16 base model loaded.")

# Freeze the layers in the base model
print("Freezing layers in the base model...")
for layer in base_model.layers:
    layer.trainable = False
print("Base model layers frozen.")

# Add a global average pooling layer
print("Adding GlobalAveragePooling2D layer...")
x = base_model.output
x = GlobalAveragePooling2D()(x)
print("GlobalAveragePooling2D layer added.")

# Add a fully connected layer with 256 units and ReLU activation
print("Adding Dense layer (256 units, ReLU)...")
x = Dense(256, activation=\'relu\')(x)
print("Dense layer (256 units, ReLU) added.")

# Add the final classification layer with 10 units (for CIFAR-10 classes) and softmax activation
print("Adding classification layer (10 units, softmax)...")
predictions = Dense(10, activation=\'softmax\')(x)
print("Classification layer added.")

# Create the fine-tuned model
print("Creating the fine-tuned model...")
model = Model(inputs=base_model.input, outputs=predictions)
print("Fine-tuned model created.")

model.summary() # Print model summary to verify architecture

