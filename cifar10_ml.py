"""
cifar10_ml.py usage

----------------------------------------------------------------------------------------------

## 0. INITIAL IMPORT STATEMENT PACKAGE NAMES:

1. NumPy for N-Dimensional Arrays
2. PyTorch - Torchvision: https://docs.pytorch.org/vision/stable/index.html
    a. Datasets Class: https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CIFAR10.html
    b. Transforms Class: https://docs.pytorch.org/vision/stable/transforms.html
3. Matplotlib
    a. Pyplot Class: https://matplotlib.org/stable/api/pyplot_summary.html
4. Seaborn for Heatmaps: https://seaborn.pydata.org/generated/seaborn.heatmap.html
5. Scikitlearn for Classifier Training and Accuracy Reports:
    a. Metrics Class: classification_report and confusion_matrix function
    b. Neighbors Class: KNeighborsClassifier
    c. Tree Class: DecisionTreeClassifier
    d. Ensemble Class: RandomForestClassifier
    e. SVM Class: C-Support Vector Classification 
    f. Neural_Network Class: Multi-layer Perceptron Classification https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
6. Time Module: https://docs.python.org/3/library/time.html


## 1. LOADING CIFAR-10 USING TORCHVISION

# Documentation: https://docs.pytorch.org/vision/master/generated/torchvision.transforms.ToTensor.html

The transform converts each PIL image into a torch.Tensor and normalizes the pixel values:

Type:           PIL.Image.Image -> torch.Tensor
Shape:          (32, 32, 3)     -> (3, 32, 32)
Pixel Range:    [0, 255](uint8) -> [0.0, 1.0] (float32)
Layout:         H x W x C       -> C x H x W

Format is compatible with PyTorch models.

# For this implementation assume these Type, Shape, Pixel Range, and Layout for image preprocessing.


## 2. Convert to NumPy arrays and preprocessing

# data_to_numpy usage:

1. Convert Images to Flatten NumPy Arrays (X)
    - Images are torch.Tensor with shape (C, H, W) = (3, 32, 32)
    - Change shape to (H, W, C) = (32, 32, 3) to match standard image layout
    - .flatten() turns the image into a 1D array of shape (3072,)   aside: 32 x 32 x 3 = 3072

2. Extracts Class Labels into a NumPy Array (y)
    - Grab integer label for each image and store into a simple list converted to a NumPy array


## 3. Generate Report
## 4. def evaluate_model(model, model_name)
"""

import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time as time


class CIFAR10Evaluator:
    def __init__(self):
        """ Initialize training and testing split """
        self.transform = transforms.ToTensor()
        self.train_data = CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.test_data = CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        self.X_train, self.y_train = self.data_to_numpy(self.train_data)
        self.X_test, self.y_test = self.data_to_numpy(self.test_data)

    def data_to_numpy(self, dataset):
        X = np.array([img.permute(1, 2, 0).flatten().numpy() for img, _ in dataset])
        y = np.array([label for _, label in dataset])
        return X, y

    def evaluate_model(self, model, model_name):
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        duration = time.time() - start_time
        y_prediction = model.predict(self.X_test)

        print(f"{model_name} training time: {duration:.2f} seconds.")
        print(f"\n{model_name} Classification Report:")
        print(classification_report(self.y_test, y_prediction))
        print("-----------------------------------------------------")

        self.plot_confusion_matrix(model_name, self.y_test, y_prediction)

    def plot_confusion_matrix(self, model_name, y_true, y_prediction):
        cm = confusion_matrix(y_true, y_prediction)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd')
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()


def main():
    """ Create evaluator object and evaluate each model """
    evaluator = CIFAR10Evaluator()

    models = [
        (KNeighborsClassifier(n_neighbors=3), "K-Nearest Neighbors"),
        (DecisionTreeClassifier(max_depth=20), "Decision Tree"),
        (RandomForestClassifier(n_estimators=50, max_depth=20), "Random Forest"),
        (SVC(kernel='rbf', C=1), "Support Vector Classifier (SVC)"),
        (MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=10), "Multilayer perceptron (MLP)")
    ]

    for model, model_name in models:
        evaluator.evaluate_model(model, model_name)


if __name__=="__main__":
    main()
