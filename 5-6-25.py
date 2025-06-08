# Import necessary libraries
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load the Wine dataset
data = load_wine()
X = data.data
y = data.target

# Initialize the KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=10000, random_state=42)

# List to store accuracy scores for each fold
fold_accuracies = []

# Perform manual 5-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    # Split the data into training and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy for the current fold
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)
    
    # Print accuracy for the current fold
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")

# Calculate and print the average accuracy across all folds
average_accuracy = np.mean(fold_accuracies)
print(f"\nAverage Accuracy across all folds: {average_accuracy:.4f}")
