from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from utility import load_malicious_benign_data

data = load_malicious_benign_data()
print("Dataframe size:",data.size)

# Split features and target variable
X = data.drop(columns=['SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'NUMBER_SPECIAL_CHARACTERS', 'CHARSET',
                       'TCP_CONVERSATION_EXCHANGE','DIST_REMOTE_TCP_PORT', 'REMOTE_IPS', 'APP_BYTES',
                       'SOURCE_APP_PACKETS', 'REMOTE_APP_PACKETS','APP_PACKETS','WHOIS_REGDATE', 'WHOIS_UPDATED_DATE'])  # Features
y = data['Type']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
print("Test Accuracy:", test_accuracy)
