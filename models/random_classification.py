import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from utility import (load_random_generator,evaluate_decision_tree,max_depths, max_features_list,
                     min_samples_splits,min_samples_leafs,max_leaf_nodes_list,
                     plot_decision_boundaries,plot_graph_on_quality_metrics)

data = load_random_generator()

X, y = data["X"], data["Y"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Test the function with different hyperparameters
best_accuracy = -1
best_hyperparameters = {}

for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        for min_samples_leaf in min_samples_leafs:
            for max_leaf_nodes in max_leaf_nodes_list:
                for max_features in max_features_list:
                    accuracy = evaluate_decision_tree(max_depth, min_samples_split, min_samples_leaf,
                                                      max_leaf_nodes, max_features,X_train, y_train)

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
print("Train Accuracy:", best_accuracy)

# Predictions
train_predictions = best_clf.predict(X_train)
test_predictions = best_clf.predict(X_test)
print(f"Train Predictions: {train_predictions} \nTest Predictions: {test_predictions}")

# Assuming best_clf is the trained model, Display graphical representation of the decision boundaries
plot_decision_boundaries(X_test, y_test, best_clf)

print("quality metrics:")
plot_graph_on_quality_metrics(y_train,y_test,train_predictions,test_predictions)