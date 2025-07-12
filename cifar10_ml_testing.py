"""
cifar10_ml_test.py usage

Testing script to evaluate model performance based on a limited set of parameters
"""

# Import statements

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time as time

# 1. Loading CIFAR-10 using torchvision
"""
Documentation: https://docs.pytorch.org/vision/master/generated/torchvision.transforms.ToTensor.html

The transform converts each PIL image into a torch.Tensor and normalizes the pixel values:

Type:           PIL.Image.Image -> torch.Tensor
Shape:          (32, 32, 3)     -> (3, 32, 32)
Pixel Range:    [0, 255](uint8) -> [0.0, 1.0] (float32)
Layout:         H x W x C       -> C x H x W

Format is compatible with PyTorch models.
"""
transform = transforms.Compose([transforms.ToTensor()])
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# 2. Convert to NumPy arrays and preprocessing
"""
data_to_numpy usage:

1. Convert Images to Flatten NumPy Arrays (X)
    - Images are torch.Tensor with shape (C, H, W) = (3, 32, 32)
    - Change shape to (H, W, C) = (32, 32, 3) to match standard image layout 
    - .flatten() turns the image into a 1D array of shape (3072,)   aside: 32 x 32 x 3 = 3072
    
2. Extracts Class Labels into a NumPy Array (y)
    - Grab integer label for each image and store into a simple list converted to a NumPy array
"""
def data_to_numpy(dataset):
    X = np.array([np.array(img.permute(1,2,0)).flatten() for img, _ in dataset])
    y = np.array([label for _, label in dataset])
    return X, y

X_train, y_train = data_to_numpy(trainset)
X_test, y_test = data_to_numpy(testset)

# Subsample for testing
X_train_mini, y_train_mini = X_train[:10000], y_train[:10000]
X_test_mini, y_test_mini = X_test[:2000], y_test[:2000]

# 3. Define evaluation function
"""
evaluate_model_testing usage:

1. Evaluate models accuracy based on classifications report and confusion matrix image
"""
def evaluate_model_testing(model, name):
    print(f"\nTraining {name}...")

    # Generate a start time and end time and find out how long it took to train the model
    start_time = time.time()
    model.fit(X_train_mini, y_train_mini)
    duration = time.time() - start_time

    y_pred = model.predict(X_test_mini)
    print(f"{name} training time: {duration:.2f} seconds.")

    # Generate a classification report
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test_mini, y_pred))

    print("-----------------------------------------------------")

    # Generate a confusion_matrix
    cm = confusion_matrix(y_test_mini, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PuRd')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    return model # Avoid passing None into argument

# 4. Define show_misclassifications function
"""
show_misclassifications usage:

1. Demonstrate instances where certain models generated incorrect classification for amusement and analysis
"""
def show_misclassifications(model, X, y, class_names, count=5):
    preds = model.predict(X)
    wrong = np.where(preds != y)[0]
    if len(wrong) == 0:
        print("No miss-classifications in sample!")
        return
    indices = np.random.choice(wrong, min(count, len(wrong)), replace=False)

    plt.figure(figsize=(count * 2, 2))
    for i, idx in enumerate(indices):
        img = X[idx].reshape(32, 32, 3)
        plt.subplot(1, count, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"T: {class_names[y[idx]]}\nP: {class_names[preds[idx]]}")
    plt.tight_layout()
    plt.show()



# 5. Train, evaluate, and show mis-classifications
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

models = [
    (KNeighborsClassifier(n_neighbors=3),                     "K-Nearest Neighbors"),
    (DecisionTreeClassifier(max_depth=20),                    "Decision Tree"),
    (RandomForestClassifier(n_estimators=50, max_depth=20),   "Random Forest"),
    (SVC(kernel='rbf', C=1),                                  "Support Vector Classifier (SVC)"),
    (MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50),"Multilayer Perceptron (MLP)")
]

for m, name in models:
    trained_model = evaluate_model_testing(m, name)
    show_misclassifications(trained_model, X_test_mini, y_test_mini,
                            class_names, count=5)

