import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

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
