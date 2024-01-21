# Decision Tree Implementation

This project consists of a custom implementation of a decision tree classifier in Python. The decision tree is designed for both classification tasks and can handle both categorical and numerical features. The implementation includes training the decision tree on a given dataset and making predictions.

## Table of Contents
1. [Decision Tree Implementation](#decision-tree-implementation)
2. [Usage](#usage)
3. [Implementation Details](#implementation-details)
4. [Example Usage with Iris Dataset](#example-usage-with-iris-dataset)
5. [Example Usage with Diabetes Dataset](#example-usage-with-diabetes-dataset)
6. [Example Usage with Imputed Diabetes Dataset](#example-usage-with-imputed-diabetes-dataset)

## Usage

To use the decision tree, follow these steps:

1. Import the `DecisionTree` class from the provided `Code` module:

   ```python
   from Code import DecisionTree
   ```

2. Create an instance of the `DecisionTree` class, specifying optional parameters such as `min_split`, `max_depth`, and `num_features`.

   ```python
   clf = DecisionTree(min_split=4, max_depth=50, num_features=None)
   ```

3. Train the decision tree using your dataset:

   ```python
   clf.fit(X_train, y_train)
   ```

4. Make predictions on a test dataset:

   ```python
   predictions = clf.predict(X_test)
   ```

5. Evaluate the model using metrics such as accuracy or confusion matrix.

## Implementation Details

### DecisionTree Class

The `DecisionTree` class contains the following methods:

- `fit(X, y)`: Trains the decision tree on the provided training data (`X` features and `y` labels).
- `predict(X)`: Predicts the class labels for a set of input samples using the trained decision tree.

### Node Class

The `Node` class represents a node in the decision tree and contains information about the split.

### Dataset Splitting

The decision tree implementation employs a recursive approach to grow the tree, considering stopping conditions such as minimum split size, maximum depth, and homogeneity of labels.

### Example Datasets

This README includes examples of using the decision tree with the Iris dataset and the Diabetes dataset. It also demonstrates how to handle missing values by imputing them using `IterativeImputer` from scikit-learn.

## Example Usage with Iris Dataset

The provided example shows how to use the decision tree with the Iris dataset, split into training and testing sets. The accuracy and confusion matrix are calculated to evaluate the model's performance.

## Example Usage with Diabetes Dataset

The README provides an example of using the decision tree with the Diabetes dataset, including dataset splitting, training, prediction, and evaluation using accuracy and a confusion matrix.

## Example Usage with Imputed Diabetes Dataset

This example extends the usage to handle missing values by imputing them using `IterativeImputer` from scikit-learn. The decision tree is trained on the imputed dataset, and predictions, a confusion matrix, and accuracy are computed for evaluation.

Feel free to adapt the provided code for your specific use case and datasets.

Abo Rabia Rami
