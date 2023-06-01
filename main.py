import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import classification_report
import graphviz
from utils import find_best_attribute

# Load and preprocess the training data
data = pd.read_csv('training_dataset.csv')

# Finding the first best attribute to split on
# This is for illustrative purposes, as the sklearn DecisionTreeClassifier will determine the best splits internally.
best_split = find_best_attribute(data, 'willwait')
print("Best Attribute to split on: ", best_split)

# Initialize LabelEncoder...
# LabelEncoder is a utility class to help normalize labels such that they contain only values between 0 and n_classes-1
le = preprocessing.LabelEncoder()

# Converts all categorical columns to numeric...
# By iterating over each column, apply the label encoding, and assign the result back to the column
for col in data.columns:
    data[col] = le.fit_transform(data[col])

# Separate feature columns and target column
feature_columns = data.columns[:-1]  # all columns except the last one
target_column = data.columns[-1]  # the last column

# Splits dataset into feature vectors (X) and target variable (y)
X = data[feature_columns]
y = data[target_column]

# Train the classifier to build the tree (using entropy as the criterion)
dtc = tree.DecisionTreeClassifier(criterion="entropy")
dtc.fit(X, y)

# Evaluate the classifier's performance on the training data
y_pred = dtc.predict(X)
print(classification_report(y, y_pred))

# Visualize the decision tree using Graphviz
# export_graphviz exports a decision tree in DOT format
dot_data = tree.export_graphviz(dtc, out_file=None,
                                feature_names=data.columns[:-1],
                                class_names=le.classes_,
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_training_data")

# Prompt the user for the test data
test_data_filename = input("Enter the name of the test data file (include .csv): ")

# Load and preprocess the test data
test_data = pd.read_csv(test_data_filename)
for col in test_data.columns:
    test_data[col] = le.fit_transform(test_data[col])

# Finding the first best attribute to split on for test data
# This is for illustrative purposes, as the sklearn DecisionTreeClassifier will determine the best splits internally.
best_split_test = find_best_attribute(test_data, 'willwait')
print("Best Attribute to split on: ", best_split_test)

# Split the test dataset into feature vectors (X_test) and target variable (y_test)
X_test = test_data.drop(columns='willwait')
y_test = test_data['willwait']

# Evaluate the classifier's performance on the test data
y_test_pred = dtc.predict(X_test)
print(classification_report(y_test, y_test_pred))

# Visualize the decision tree for the test data
dot_data = tree.export_graphviz(dtc, out_file=None,
                                feature_names=test_data.columns[:-1],
                                class_names=le.classes_,
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_test_data")
