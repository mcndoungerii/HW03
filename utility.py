import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

# Train decision tree classifiers with different hyperparameters
max_depths: list = [None, 5, 10]
min_samples_splits: list = [2, 5, 10]
min_samples_leafs: list = [1, 2, 5]
max_leaf_nodes_list: list = [None, 10, 20]
max_features_list: list = [None, 'sqrt', 'log2']


def load_malicious_benign_data():

    # Load the dataset
    df = pd.read_csv("../data/dataset_Malicious_and_Benign_Websites.csv")

    # Convert date columns to datetime objects
    df['WHOIS_REGDATE'] = pd.to_datetime(df['WHOIS_REGDATE'], errors='coerce')
    df['WHOIS_UPDATED_DATE'] = pd.to_datetime(df['WHOIS_UPDATED_DATE'], errors='coerce')

    # Handle missing values
    # For now, let's fill the missing values
    data = df.dropna()

    data = data.convert_dtypes()
    # Set Date column as index
    data.set_index('URL', inplace=True)
    return data


def load_random_generator():
    # Generate input data
    np.random.seed(0)
    X = np.random.randn(300, 2)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    return dict(X=X, Y=Y)

# Define a function to train and evaluate decision tree classifiers with different hyperparameters
def evaluate_decision_tree(max_depth=None, min_samples_split=None, min_samples_leaf=None, max_leaf_nodes=None, max_features=None, X_train= None, y_train=None):
    # Initialize the decision tree classifier with specified hyperparameters
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
                                 max_features=max_features )
    # Train the classifier
    clf.fit(X_train, y_train)
    # Perform cross-validation to evaluate the model
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    # Calculate the mean cross-validation accuracy
    mean_cv_accuracy = cv_scores.mean()
    return mean_cv_accuracy


# Function to plot decision boundaries for the trained model and test dataset
def plot_decision_boundaries(X, y, clf):
    # Create a meshgrid for plotting
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the class for each mesh point
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundaries
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3)

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundaries')
    plt.show()


def plot_graph_(X_train,y_train, best_hyperparameters,feature1,feature2):
    # Fix hyperparameters
    max_depth = best_hyperparameters['max_depth']
    min_samples_split = best_hyperparameters['min_samples_split']

    # Extract the selected features
    X_train_subset = X_train[[feature1, feature2]]
    # Train a decision tree on the selected features
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    clf.fit(X_train_subset, y_train)
    # Plot decision boundaries
    plt.figure(figsize=(10, 6))
    # Plot the decision boundaries
    x_min, x_max = X_train_subset[feature1].min() - 1, X_train_subset[feature1].max() + 1
    y_min, y_max = X_train_subset[feature2].min() - 1, X_train_subset[feature2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    # Plot class points
    colors = ['red', 'blue']
    for idx, class_label in enumerate(np.unique(y_train)):
        plt.scatter(x=X_train_subset[y_train == class_label][feature1],
                    y=X_train_subset[y_train == class_label][feature2],
                    alpha=0.8,
                    c=colors[idx],
                    label=class_label)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title('Decision Boundary Visualization')
    plt.legend()
    plt.show()



