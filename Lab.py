import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

california = fetch_california_housing()
df = pd.DataFrame(data=california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target
print("First five rows of the dataset:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['MedHouseVal'], bins=30, kde=True)
plt.title('Histogram of MedHouseVal')
plt.subplot(1, 3, 2)
sns.scatterplot(x='MedInc', y='MedHouseVal', data=df)
plt.title('MedHouseVal vs MedInc')
plt.subplot(1, 3, 3)
sns.scatterplot(x='AveRooms', y='MedHouseVal', data=df)
plt.title('MedHouseVal vs AveRooms')
plt.tight_layout()
plt.show()
X = df[['MedInc']].values
y = df['MedHouseVal'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
def linear_regression_scratch(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best
theta = linear_regression_scratch(X_train, y_train)
print(f"\nScikit-learn model: Intercept = {model.intercept_}, Coefficient = {model.coef_[0]}")
print(f"From scratch: Intercept = {theta[0]}, Coefficient = {theta[1]}")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train.flatten(), y=y_train, label="Training data", alpha=0.5)
x_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', label="Regression line")
plt.xlabel('MedInc')
plt.ylabel('MedHouseVal')
plt.title('Regression Line on Training Data')
plt.legend()
plt.show()
