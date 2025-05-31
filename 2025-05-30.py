from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate k from 1 to 20
k_values = list(range(1, 21))
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)

# Find the best k
best_k = k_values[accuracies.index(max(accuracies))]
best_accuracy = max(accuracies)

# Print best k and accuracy
print(f"Best k: {best_k} with Accuracy: {best_accuracy:.2f}")

# Plot k vs. accuracy
plt.plot(k_values, accuracies, marker='o')
plt.title('k-NN Accuracy on Iris Validation Set')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
