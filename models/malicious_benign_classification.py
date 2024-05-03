import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from utility import load_malicious_benign_data,evaluate_decision_tree

data = load_malicious_benign_data()

# Split features and target variable
X = data.drop(columns=['SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'NUMBER_SPECIAL_CHARACTERS', 'CHARSET',
                       'TCP_CONVERSATION_EXCHANGE',
                       'DIST_REMOTE_TCP_PORT', 'REMOTE_IPS', 'APP_BYTES', 'SOURCE_APP_PACKETS', 'REMOTE_APP_PACKETS',
                       'APP_PACKETS',
                       'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE'])  # Features
y = data['Type']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree classifiers with different hyperparameters
max_depths = [None, 5, 10]
min_samples_splits = [2, 5, 10]
min_samples_leafs = [1, 2, 4]
max_leaf_nodes = [None, 5, 10]
max_features = [None, 'sqrt', 'log2']
best_accuracy = -1
best_hyperparameters = {}

for max_depth in [None, 5, 10]:
    for min_samples_split in [2, 5, 10]:
        for min_samples_leaf in [1, 2, 5]:
            for max_leaf_nodes in [None, 10, 20]:
                for max_features in [None, 'sqrt', 'log2']:
                    accuracy = evaluate_decision_tree(max_depth, min_samples_split, min_samples_leaf,
                                                      max_leaf_nodes, max_features, X_train, y_train)

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
