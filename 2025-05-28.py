import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Data Loading and Cleaning
df = pd.read_csv('student-mat.csv', sep=',', quotechar='"')
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# Convert categorical columns to numeric (one-hot encoding)
df = pd.get_dummies(df, drop_first=True)

# Task 2: Feature Selection & Engineering
df['avg_grade'] = (df['G1'] + df['G2']) / 2
df['engagement_score'] = df['studytime'] - df['goout']

# Remove outliers in 'absences' and 'failures'
for col in ['absences', 'failures']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# Final features
features = ['avg_grade', 'engagement_score']
target = 'G3'
X = df[features].values
y = df[target].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 3: Linear Regression from Scratch
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            dw = (1/n_samples) * np.dot(X.T, error)
            db = (1/n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            loss = (1/(2*n_samples)) * np.sum(error**2)
            self.loss_history.append(loss)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train from scratch
scratch_model = LinearRegressionScratch()
scratch_model.fit(X_train, y_train)
y_pred_scratch = scratch_model.predict(X_test)

# Plot loss
plt.plot(scratch_model.loss_history)
plt.title("Training Loss Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

# Task 4: Linear Regression using scikit-learn
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_sklearn = model.predict(X_test)

print("Scikit-learn Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Task 5: Visualization of Predictions
plt.figure(figsize=(10,5))

# Scatter actual vs predicted
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_sklearn)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted (Linear)")

# Residual plot
plt.subplot(1, 2, 2)
sns.residplot(x=y_test, y=y_pred_sklearn, lowess=True)
plt.xlabel("Actual G3")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()
plt.show()

# Task 6: Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = Line_
