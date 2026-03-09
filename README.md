# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Dataset Loading & Preparation Load the Iris dataset using load_iris(). Convert it into a Pandas DataFrame. Separate features (X) and target labels (y).
2.Split the dataset into training and testing sets using train_test_split(). Use 80% data for training and 20% for testing. 3.Create an SGDClassifier with specified parameters. Train the classifier using the training data (fit() method).
3.Predict class labels for test data. Calculate accuracy using accuracy_score(). Generate the confusion matrix to evaluate classification performance.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Dharshini k
RegisterNumber: 25004639
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Iris dataset
iris = load_iris()
# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
# Display the first few rows of the dataset
print(df.head())
# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)
# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)  

*/
```
## Output:
![prediction of iris species using SGD Classifier](sam.png)
<img width="725" height="350" alt="image" src="https://github.com/user-attachments/assets/c7ebbe85-64aa-4a6a-a172-04c560739024" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
