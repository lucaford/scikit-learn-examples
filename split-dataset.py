from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# load iris dataset
iris = datasets.load_iris()

# split data in features(X) and labels(y)
X = iris.data
y = iris.target

# split train and test variables from dataset. We're getting 20% of dataset for test and remaning for training

X_train, X_test, y_train, y_text = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_text.shape)
