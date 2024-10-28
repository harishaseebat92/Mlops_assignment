import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import missingno as msno

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
df
print(df.head())
print(df.describe())

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

print(df.isnull().sum())

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
# Saving the heatmap image
plt.savefig('correlation_heatmap.png') # Save heatmap as an image
plt.show()


sns.scatterplot(x='rm', y='medv', size= 0.1, data=df)
plt.title("Relationship between 'rm' and 'medv'")
plt.xlabel("Average Number of Rooms (rm)")
plt.ylabel("Median Value of Homes (medv)")
plt.show()

# Write your code after reading about data splitting using train_test_split and implement it here.

X_train, X_test, y_train, y_test = train_test_split(
    df[["rm"]], df["medv"], test_size=0.33, random_state=42)


# Write your code for implementing the Linear Regression model
model = LinearRegression()

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Write your code for evaluating model performance

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Assuming predictions are made (y_pred) and actual data is y_test
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('House Price')
plt.legend()
plt.show()