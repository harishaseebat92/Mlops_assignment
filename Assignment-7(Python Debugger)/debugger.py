from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
import pdb
def train_model(X, y):
    model = LinearRegression()
   #  pdb.set_trace() # Debugger for model training
    model.fit(X, y)
    return model
# Example Data
X = np.array([[1], [2], [3], [4]])
y = np.array([1, 4, 9, 16, 25]) # Incorrect shape on purpose
y = y[:X.shape[0]]  # Slice y to keep the first 4 elements

trained_model = train_model(X, y)
print("Trained Model Coefficients:", trained_model.coef_)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    pdb.set_trace() # Debugger for evaluation
    mse = mean_squared_error(y_test, predictions)
    return mse
X_test = np.array([[5], [6], [7], [8]])
y_test = np.array([25, 36, 49, 64])
# Using trained_model from previous task
mse_score = evaluate_model(trained_model, X_test, y_test)
print("Mean Squared Error:", mse_score)

# (Pdb) pp predictions
# array([20., 25., 30., 35.])

# (Pdb) pp y_test
# array([25, 36, 49, 64])