from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

data = fetch_california_housing()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

print("All features:", data.feature_names)

correlation = df.corr()['target'].abs().sort_values(ascending=False)
top_features = correlation.index[1:5]  
print("\nTop 4 features:", top_features.tolist())

X = df[top_features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("\nModel coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R² Score:", r2)
