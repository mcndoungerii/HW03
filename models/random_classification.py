import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from utility import load_random_generator
data = load_random_generator()

X, y = data["X"], data["Y"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Define a function to train and evaluate decision tree classifiers with different hyperparameters
def evaluate_decision_tree(max_depth=None, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, max_features=None):
    # Initialize the decision tree classifier with specified hyperparameters
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
                                  max_features=max_features)
    # Train the classifier
    clf.fit(X_train, y_train)
    # Perform cross-validation to evaluate the model
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    # Calculate the mean cross-validation accuracy
    mean_cv_accuracy = cv_scores.mean()
    return mean_cv_accuracy

# Test the function with different hyperparameters
best_accuracy = -1
best_hyperparameters = {}

for max_depth in [None, 5, 10]:
    for min_samples_split in [2, 5, 10]:
        for min_samples_leaf in [1, 2, 5]:
            for max_leaf_nodes in [None, 10, 20]:
                for max_features in [None, 'sqrt', 'log2']:
                    accuracy = evaluate_decision_tree(max_depth, min_samples_split, min_samples_leaf,
                                                      max_leaf_nodes, max_features)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_hyperparameters = {
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'max_leaf_nodes': max_leaf_nodes,
                            'max_features': max_features
                        }

# Train the best model on the full training set
best_clf = DecisionTreeClassifier(**best_hyperparameters)
best_clf.fit(X_train, y_train)

# Evaluate the best model on the testing set
y_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Best Model Hyperparameters:", best_hyperparameters)
print("Test Accuracy:", test_accuracy)
