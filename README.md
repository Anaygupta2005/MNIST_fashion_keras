# Fashion-MNIST Apparel Classifier

This project implements a **fully connected neural network** in TensorFlow/Keras to classify grayscale 28x28 images of clothing items from the Fashion-MNIST dataset. It includes data visualization, model training, performance evaluation, and visualization of misclassified examples.

> Final model achieves **97% accuracy on the test set**.

## Dataset

Fashion-MNIST is a dataset of 70,000 grayscale images categorized into 10 clothing classes. Each image is 28x28 pixels and falls under one of the following categories:

| Label | Class        |
|-------|--------------|
| 0     | T-shirt/top  |
| 1     | Trouser      |
| 2     | Pullover     |
| 3     | Dress        |
| 4     | Coat         |
| 5     | Sandal       |
| 6     | Shirt        |
| 7     | Sneaker      |
| 8     | Bag          |
| 9     | Ankle boot   |

## Project Workflow

### 1. Data Preparation
- The dataset is loaded using `tf.keras.datasets.fashion_mnist`.
- Images are flattened to vectors of length 784 (28x28).
- The training and testing data are converted to pandas DataFrames.
- A CSV export is created (`fashion.csv`) which includes labels and pixel values.

### 2. Data Exploration
- Class distribution is visualized using a seaborn countplot.
- 25 random samples are displayed in a 5x5 grid with their labels.
