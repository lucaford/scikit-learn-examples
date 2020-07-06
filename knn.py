import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# read dataset from https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

data = pd.read_csv('./datasets/car.data')

# split data into features and label

X = data[[
    'buying',
    'maint',
    'safety'
]].values
y = data[['class']]

# we have to convert data so it can be read (to numbers). First we convert X using LabelEncoder (one way)

Le = LabelEncoder()

for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])


# then we convert y using mapping (another way)

label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)

print(y)

# create knn model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

# split data in trainning data and testing data (20% of data is for testing)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model

knn.fit(X_train, y_train)

# make predictions with X_test (data to test)

prediction = knn.predict(X_test)

# show prediction accuracy with y_test (result)

accuracy = metrics.accuracy_score(y_test, prediction)

print("Predicitions: ", prediction)
print("Accuracy: ", accuracy)

# Predict actual value and see how our model works!

print("Actual value: ", y[26])
print("Prediction value: ", knn.predict(X)[26])
