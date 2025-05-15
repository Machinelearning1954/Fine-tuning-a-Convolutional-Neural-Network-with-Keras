# Fine-Tuning a Pre-trained CNN (VGG16) on CIFAR-10

## 1. Introduction

This project demonstrates the process of fine-tuning a pre-trained Convolutional Neural Network (CNN), specifically VGG16, for a new image classification task using the CIFAR-10 dataset. Transfer learning is employed, leveraging the knowledge learned by VGG16 on the large ImageNet dataset to improve performance and reduce training time on the target task.

The key steps involved are:
- Loading and preprocessing the CIFAR-10 dataset.
- Loading the pre-trained VGG16 model, excluding its top classification layer.
- Freezing the weights of the base VGG16 model.
- Adding new custom classification layers on top of the VGG16 base.
- Training the modified model on the CIFAR-10 dataset.
- Evaluating the performance of the fine-tuned model.

## 2. Dataset: CIFAR-10

The CIFAR-10 dataset is a widely used benchmark for computer vision tasks. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is split into 50,000 training images and 10,000 test images.

## 3. Methodology

### 3.1. Data Preprocessing

The CIFAR-10 images and labels are preprocessed as follows:
- **Loading Data**: The dataset is loaded using `tensorflow.keras.datasets.cifar10`.
- **Normalization**: Pixel values of the images are normalized from the range [0, 255] to [0, 1] by dividing by 255.0. This helps in stabilizing the training process.
- **One-Hot Encoding**: The integer labels (0-9) are converted into one-hot encoded vectors. For example, class '3' becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.
- **Train-Validation Split**: The original training data (50,000 images) is further split into a training set (90%, i.e., 45,000 images) and a validation set (10%, i.e., 5,000 images). The validation set is used to monitor the model's performance during training and for early stopping.

### 3.2. Model Architecture: Transfer Learning with VGG16

- **Base Model**: The VGG16 model, pre-trained on ImageNet, is used as the base feature extractor. The `include_top=False` argument is used to exclude the original fully connected classification layers of VGG16, as we are adapting it for CIFAR-10 which has 10 classes (ImageNet has 1000). The input shape is set to `(32, 32, 3)` corresponding to CIFAR-10 image dimensions.
- **Freezing Base Layers**: The weights of all layers in the VGG16 base model are frozen (`layer.trainable = False`). This ensures that their learned features from ImageNet are preserved and not updated during the initial phase of fine-tuning on the smaller CIFAR-10 dataset.
- **Custom Top Layers**: New layers are added on top of the frozen VGG16 base:
    1.  `GlobalAveragePooling2D`: This layer flattens the output of the VGG16 base by taking the average of each feature map.
    2.  `Dense` layer: A fully connected layer with 256 units and ReLU activation function.
    3.  `Dense` (Output) layer: The final classification layer with 10 units (one for each CIFAR-10 class) and a `softmax` activation function to output probabilities for each class.
- **Final Model**: The new model is constructed by combining the VGG16 base input with the custom prediction layers.

### 3.3. Training

- **Compilation**: The model is compiled with:
    - Optimizer: `Adam` with a learning rate of `0.001`.
    - Loss Function: `categorical_crossentropy`, suitable for multi-class classification with one-hot encoded labels.
    - Metrics: `accuracy` is monitored during training and evaluation.
- **Training Process**: The model is trained for a specified number of epochs (e.g., 10 epochs) using the preprocessed training data and validated on the validation set.
- **Early Stopping**: An `EarlyStopping` callback is used to monitor the validation loss (`val_loss`). If the validation loss does not improve for a certain number of epochs (patience, e.g., 3), training is stopped, and the weights from the epoch with the best validation loss are restored. This helps prevent overfitting.

## 4. Results

The model is evaluated on the test set after training. The performance metrics are:
- **Test Loss**: [To be filled after training - e.g., 0.XX]
- **Test Accuracy**: [To be filled after training - e.g., YY.YY%]

(These values will be updated once the training script `cnn_finetune_train_eval_full.py` completes and the `training_evaluation_log.txt` is analyzed.)

## 5. How to Run

1.  **Prerequisites**: Ensure Python 3.11 and the required libraries (see `Dependencies` section) are installed.
2.  **Clone the repository** (if applicable) or ensure the script `cnn_finetune_train_eval_full.py` is available.
3.  **Run the script**:
    ```bash
    python3.11 cnn_finetune_train_eval_full.py
    ```
    This script will:
    - Download and preprocess the CIFAR-10 dataset.
    - Build the VGG16-based fine-tuned model.
    - Train the model.
    - Save the trained model to `cifar10_vgg16_finetuned.keras`.
    - Evaluate the model on the test set and print the results.
    - Training and evaluation logs will be printed to the console and also saved in `training_evaluation_log.txt` if output is redirected.

## 6. Dependencies

The project requires the following Python libraries:
- `tensorflow` (tested with version 2.19.0 or similar)
- `numpy`
- `scikit-learn` (for `train_test_split`)

These can be installed using pip:
```bash
pip3 install tensorflow scikit-learn numpy
```

## 7. File Structure

- `cnn_finetune_train_eval_full.py`: The main Python script containing all code for data loading, preprocessing, model building, training, and evaluation.
- `cifar10_vgg16_finetuned.keras`: The saved trained Keras model (after successful execution of the script).
- `training_evaluation_log.txt`: Log file containing the output of the training and evaluation process (if output is redirected).
- `README.md`: This file.
- `todo.md`: Task checklist for the project.

