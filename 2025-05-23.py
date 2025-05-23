import csv
import numpy as np

# Load CSV manually
with open('Titanic-Dataset.csv', 'r') as file:
    data = list(csv.reader(file))
    header = data[0]
    rows = data[1:]

# Extract column indices
pclass_idx = header.index('Pclass')
age_idx = header.index('Age')
sibsp_idx = header.index('SibSp')
parch_idx = header.index('Parch')
fare_idx = header.index('Fare')

# Prepare feature and target lists
X_list = []
y_list = []

for row in rows:
    try:
        pclass = float(row[pclass_idx])
        age = float(row[age_idx])
        sibsp = float(row[sibsp_idx])
        parch = float(row[parch_idx])
        fare = float(row[fare_idx])
        X_list.append([pclass, age, sibsp, parch])
        y_list.append(fare)
    except:
        continue  # skip rows with missing or invalid data

# Convert to numpy arrays
X = np.array(X_list)
y = np.array(y_list).reshape(-1, 1)

# Add bias term
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Compute weights using Normal Equation
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Print coefficients
print("Model coefficients (theta):")
print(theta)

# Prediction function
def predict(X_input):
    X_input = np.array(X_input)
    if X_input.ndim == 1:
        X_input = X_input.reshape(1, -1)
    X_input_b = np.c_[np.ones((X_input.shape[0], 1)), X_input]
    return X_input_b.dot(theta)

# Example usage
sample_input = [[3, 22, 1, 0], [1, 38, 1, 0]]
predicted_fares = predict(sample_input)
print("\nPredicted fares:")
print(predicted_fares)
