#Rami Abo Rabia, Jacob Zack

from sklearn import datasets
from sklearn.metrics import confusion_matrix
import numpy as np
from Code import DecisionTree

#Splits a dataset into training and testing sets.
def split_dataset(X, y, test_size=0.2, random_state=None): #1:4 as default.
    #test_size is the proportion of the dataset to include in the test split.
    #random_state for random number generator (for reproducibility).
    
    if random_state is not None:
        np.random.seed(random_state)

    #Calculate the number of samples in the dataset
    num_samples = len(X)
    
    #Calculate the number of samples in the test set based on the test_size
    num_test_samples = int(num_samples * test_size)
    
    #Shuffle the indices of the samples
    shuffled_indices = np.random.permutation(num_samples)

    #Split the dataset into training and testing sets based on shuffled indices
    X_train = X[shuffled_indices[:-num_test_samples]] #Training feature matrix
    X_test = X[shuffled_indices[-num_test_samples:]] #Testing feature matrix

    y_train = y[shuffled_indices[:-num_test_samples]] #Training target vector
    y_test = y[shuffled_indices[-num_test_samples:]] #Testing target vector

    #Return the split datasets
    return X_train, X_test, y_train, y_test


data = datasets.load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.33, random_state=1234)

clf = DecisionTree()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

y_pred = clf.predict(X_test)

print("Misclassified Predictions:")
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print(f"True Label: {y_test[i]}, Predicted Label: {y_pred[i]}, Data Point: {X_test[i]}")
        
# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

def accuracy(y_test, predictions):
    return np.sum(y_test == predictions) / len(predictions)

acc = accuracy(y_test, predictions)
print(acc)