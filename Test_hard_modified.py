from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix
import numpy as np
from Code import DecisionTree
import pandas as pd

def split_dataset(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = len(X)
    num_test_samples = int(num_samples * test_size)
    shuffled_indices = np.random.permutation(num_samples)

    X_train = X[shuffled_indices[:-num_test_samples]]
    X_test = X[shuffled_indices[-num_test_samples:]]

    y_train = y[shuffled_indices[:-num_test_samples]]
    y_test = y[shuffled_indices[-num_test_samples:]]

    return X_train, X_test, y_train, y_test

# Load the Diabetes dataset from scikit-learn
file_path = "diabetes.csv"
data = pd.read_csv(file_path)

# Replace zeros (except in "Pregnancies" and "Outcome") with NaN to indicate missing values
data[data.columns.difference(["Pregnancies", "Outcome"])] = data[data.columns.difference(["Pregnancies", "Outcome"])].replace(0, np.nan)

# Separate features and target
X = data.drop(columns=['Outcome']).values
y = data['Outcome'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=44)

# Impute missing values using IterativeImputer
imputer = IterativeImputer(max_iter=10, random_state=0)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train your decision tree classifier
clf = DecisionTree(max_depth=7)
clf.fit(X_train_imputed, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test_imputed)

# Calculate the accuracy of the model
def accuracy(y_test, predictions):
    return np.sum(y_test == predictions) / len(predictions)

# Print misclassified predictions
print("Misclassified Predictions:")
for i in range(len(y_test)):
    if y_test[i] != predictions[i]:
        print(f"True Label: {y_test[i]}, Predicted Label: {predictions[i]}, Data Point: {X_test_imputed[i]}")

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Calculate and print the accuracy
acc = accuracy(y_test, predictions)
print("Accuracy:", acc)
