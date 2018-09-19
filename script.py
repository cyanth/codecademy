# Load Wisconsin Breast Cancer dataset from built-in sklearn datasets

from sklearn.datasets import load_breast_cancer
breast_cancer_data = load_breast_cancer()

# Print out the first row and column names for features
print (breast_cancer_data.feature_names)
print (breast_cancer_data.data[0])

# Print out target information
print (breast_cancer_data.target)
print (breast_cancer_data.target_names)

X = breast_cancer_data.data
y = breast_cancer_data.target
    
from sklearn.model_selection import train_test_split

# Specified complementary test_size, since train_size gives deprecation warning.
# Renamed y_ instead of label. 

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=100)

print (X_train.shape, y_train.shape, X_validation.shape, y_validation.shape)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
print (classifier.score(X_validation, y_validation))

k_list = range(1,101)
accuracies = []
for i in k_list:
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_validation, y_validation)
    print (i, score)
    accuracies.append(score)

import matplotlib.pyplot as plt
import numpy as np

plt.plot(k_list, accuracies)
plt.xlabel("K")
plt.ylabel("Breast Cancer Classifier Accuracy")
plt.show()
