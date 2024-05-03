from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from utility import load_random_generator

def evaluate_model(data, test_size):
    X, y = data["X"], data["Y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # Define the parameter grid
    param_grid = {
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_leaf_nodes': [None, 10, 20],
        'max_features': [None, 'sqrt', 'log2']
    }
    # Initialize the DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    # Initialize GridSearchCV
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    # Get the best estimator
    best_clf = grid_search.best_estimator_
    # Evaluate the best model on the testing set
    y_pred = best_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Best Model Hyperparameters:", grid_search.best_params_)

    return test_accuracy

data = load_random_generator()

# Define different test sizes (volume of the sample)
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

# Evaluate model for each test size
for test_size in test_sizes:
    accuracy = evaluate_model(data, test_size)
    print(f"Test Size: {test_size}, Test Accuracy: {accuracy}")


