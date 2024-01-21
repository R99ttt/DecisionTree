#Rami Abo Rabia, Jacob Zack

import pandas as pd
import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None,threshold=None, left = None, right=None, value=None):
        self.feature = feature #The index of the feature that the node splits on.
        self.threshold = threshold #The threshold value for the feature's split.
        self.left = left
        self.right=right
        self.value = value #The predicted class value if the node is a leaf node.
    
    #True if the node is a leaf node (has a predicted class value), False otherwise:    
    def leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_split=4, max_depth=50, num_features=None):
        self.num_features=num_features #The maximum number of features to consider for each split.
        self.min_split=min_split #The minimum number of samples required to perform a split.
        self.max_depth=max_depth #The maximum depth of the decision tree.
        self.root=None #The root node of the decision tree
    
    #Fit the decision tree to the provided training data:    
    def fit(self, X, y):
        #X is the feature matrix
        #y is the target values
        self._determine_num_features(X)
        self.root = self._construct_tree(X, y)
        
        
    #to determine the number of features to consider for each split in the decision tree:
    def _determine_num_features(self, X):
        if self.num_features: 
            self.num_features = min(X.shape[1], self.num_features)
        else:
            self.num_features = X.shape[1]
    
        
    #grows the decision tree by recursively splitting nodes:          
    def _construct_tree(self, X, y, depth = 0):
        #depth is the current depth of the tree.
        num_samples = X.shape[0] #Number of samples in the dataset
        num_features = X.shape[1] #Number of features in the dataset
        num_labels = len(np.unique(y)) #Number of "unique" target labels (classes)
        
        #stop condition
        #Check for stopping conditions to create leaf nodes:
        if(num_labels ==1 or depth>= self.max_depth or num_samples < self.min_split):
            leaf_val = self._the_common_label(y)
            return Node(value=leaf_val)
        
        #Randomly select a subset of features for potential splits:
        selected_feature_indices = np.random.choice(num_features, self.num_features, replace= False)
        
        
        #best split
        #Find the best split threshold and feature based on information gain:
        best_threshold, best_feature = self._best_split(X, y, selected_feature_indices)
        
        #child node creation:
        # Create child nodes by splitting the data based on the best split:
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        
        left = self._construct_tree(X[left_indices, :], y[left_indices], depth+1)
        right = self._construct_tree(X[right_indices, :], y[right_indices], depth+1)
        
        #Create and return the current node with the best split information:
        return Node(best_feature, best_threshold, left, right, )
        
    #Find the best split threshold and feature based on maximum information gain:    
    def _best_split(self, X, y, feature_idxs):
        #feature_idxs is a subset of feature indices to consider for splitting.
        max_gain=-1 #Initialize the maximum information gain
        split_idx = None #Index of the feature with the best split
        split_threshold = None  #Optimal threshold for the split (that achieves the maximum information gain)
        
        #Run through the selected feature indices:
        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx] #Extract the current feature column
            all_thresholds = np.unique(X_column) #Find unique values as potential thresholds
            
            #Run through all potential thresholds for the current feature
            for thr in all_thresholds: 
                #information gain calculation:
                gain = self._information_gain(X_column, y, thr) #Calculate the information gain achieved by splitting at the current threshold
                
                #Update the maximum gain and best split information if a better split is found:
                if gain > max_gain:
                    max_gain = gain
                    split_idx = feature_idx
                    split_threshold = thr
                    
        return split_threshold, split_idx
    
    #Calculate the information gain achieved by splitting the data based on a given threshold:
    def _information_gain(self, X_column, y, threshold):
        #X_column is the feature values of a single column from the feature matrix.
            
        parent_entropy = self._entropy(y) #Calculate the entropy of the parent node
        
        #left and right node creation:
        #Split the data into left and right nodes based on the threshold
        left_indices, right_indices = self._split(X_column, threshold)
        
        # If either left or right node is empty, the information gain is 0:
        if len(left_indices)==0 or len(right_indices)==0:
            return 0
        
        #Calculate the weighted entropy of the left and right subtrees
        samples_num = len(y)
        
        samples_num_left= len(left_indices)
        samples_num_right= len(right_indices)
        
        entropy_left=self._entropy(y[left_indices])
        entropy_right=self._entropy(y[right_indices])
        
        child_entropy = (samples_num_left / samples_num) * entropy_left + (samples_num_right / samples_num) * entropy_right
        
        #finally we got all the needed calculations to return the information gain, by subtracting child entropy from parent entropy:
        information_gain = parent_entropy - child_entropy
        
        return information_gain
    
    #Split the data into two subsets based on a given threshold:
    def _split(self, X_column, split_thr):
        #split_thr is the threshold value for splitting the data.
        
        left_indices = np.where(X_column <= split_thr)[0] #Indices of samples that satisfy the split condition (values <= split_thr).
        right_indices = np.where(X_column > split_thr)[0] #Indices of samples that do not satisfy the split condition (values > split_thr).
        return left_indices, right_indices
    
    #Calculate the entropy of a target variable:
    def _entropy(self, y):
        #y is an array containing the target labels.
        
        occurrence= np.bincount(y) #Count the occurrences of each unique label.
        class_probabilities = occurrence / len(y) #Calculate the probabilities of each label.
        
        entropy = 0.0 #Initialized entropy
        
        for p_x in class_probabilities:
            if p_x > 0:
                entropy -= p_x * np.log2(p_x)
        
        return entropy
    #Find the most common label in an array of target labels.
    def _the_common_label(self, y):
        counter = Counter(y) #Count occurrences of each unique label
        value = counter.most_common(1)[0][0] #Get the most common label
        
        return value #value is the most common label.
    
    #Predict the class labels for a set of input samples using the trained decision tree.
    def predict(self, X):
        #X is an array where each row represents a sample and each column represents a feature.
        
        traverse_results = [] # used to store the predicted labels for each sample in X
        
        for x in X:
            result = self._traverse(x, self.root) #Traverse the decision tree for each sample
            traverse_results.append(result) #Store the predicted label
        
        traverse_results = np.array(traverse_results) #Convert the list to a numpy array
        return traverse_results
    
    #Traverse the decision tree recursively to predict the class label for a given input sample:    
    def _traverse(self, x, node):
        #node is the current node in the decision tree being considered.
        
        #if the current node is a leaf node, return the predicted class label
        if node.leaf():
            return node.value
        
        #Determine which child node to traverse based on the feature and threshold at the current node:
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        
        return self._traverse(x, node.right) #the predicted class label for the input sample.     