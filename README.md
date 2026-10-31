# Traffic Sign Classification using CNN

## Overview

This project focuses on the implementation and training of a Convolutional Neural Network (CNN) for image classification, with the goal of recognizing and categorizing road signs.

The model is developed using **PyTorch** and trained on a labeled image dataset. Each image is classified into one of four categories, representing different road sign types.

The project covers the complete machine learning workflow, from image preprocessing and dataset preparation to model development, training, validation, and performance evaluation.

## Classification Categories

The model is designed to distinguish between the following four classes:

| Label | Description |
|---|---|
| `nothing` | No relevant road sign |
| `roundabout` | Roundabout sign |
| `stop` | Stop sign |
| `priority` | Priority road sign |

## Model Architecture

A custom CNN architecture (`SimpleCNN`) is implemented using PyTorch.

The network consists of:

- **Convolutional Layers:** Three convolutional layers with 32, 64, and 128 filters respectively, used for extracting increasingly complex visual features.
- **Batch Normalization:** Applied after each convolutional layer to improve training stability.
- **ReLU Activation:** Introduces non-linearity into the network.
- **Max Pooling:** Reduces the spatial dimensions of feature maps.
- **Fully Connected Layers:** Transform the extracted features into class predictions.
- **Dropout:** Applied with a probability of 0.5 to help reduce overfitting.

The final layer produces four output values, corresponding to the four classification categories.

## Data Preprocessing

Before being fed into the network, the images undergo the following preprocessing steps:

1. **Image Loading:** Images are loaded using OpenCV.
2. **Color Conversion:** Images are converted from BGR to RGB format.
3. **Resizing:** All images are resized to 128 × 128 pixels.
4. **Tensor Conversion:** Images are converted into PyTorch tensors with dimensions `(channels, height, width)`.
5. **Normalization:** Pixel values are normalized to the range [-1, 1].

The corresponding class labels are extracted from individual JSON annotation files and encoded into numerical values.

## Training Configuration

The model is trained using the following hyperparameters:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.00005 |
| Loss Function | Cross Entropy Loss |
| Batch Size | 32 |
| Number of Epochs | 30 |
| Input Image Size | 128 × 128 × 3 |
| Number of Classes | 4 |

Additionally, a weighted Cross Entropy Loss configuration is explored to account for potential class imbalance.

## Training & Evaluation

The training process follows the standard supervised learning workflow:

1. Forward propagation through the CNN.
2. Calculation of the loss using Cross Entropy Loss.
3. Backpropagation to compute gradients.
4. Weight updates using the Adam optimizer.

Model performance is monitored on both training and validation data throughout the training process.

The notebook also includes an evaluation stage on a separate test dataset.

The following evaluation metrics are calculated:

- **Accuracy:** Overall percentage of correctly classified images.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual instances correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Visualization of correct and incorrect predictions across all classes.

Training and validation loss and accuracy curves are also generated to monitor the learning process and assess model generalization.

## Model Saving

The trained model weights are saved in `.pth` format.

The notebook includes functionality for saving the model parameters and reloading the best-performing model based on validation loss.

## Technologies & Libraries

- Python
- PyTorch
- Torchvision
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

## Execution

The notebook is designed to run in **Google Colab**.

To execute the project:

1. Open `RASLabs2.ipynb` in Google Colab.
2. Mount Google Drive to access the image datasets.
3. Place the training and test datasets in the expected directory structure.
4. Execute the notebook cells sequentially to preprocess the data, initialize the CNN, train the model, and evaluate its performance.

### Dataset Structure

Each dataset directory is expected to contain image files (`.png`) and their corresponding annotation files (`.json`).

```text
dataset/
├── image_001.png
├── image_001.json
├── image_002.png
├── image_002.json
└── ...
```

The JSON annotation files contain the corresponding class labels under the `name` field.

**Note:** The dataset is not included in this repository. The Google Drive paths used in the notebook should be adjusted according to your local dataset location.

---

## Project Scope

This project provided practical experience with:

- Convolutional Neural Networks and deep learning fundamentals.
- Image preprocessing and computer vision techniques.
- Supervised multi-class image classification.
- Model training, validation, and hyperparameter configuration.
- Performance evaluation using classification metrics.
- Implementation of deep learning models using PyTorch.
