import pdb
import numpy as np
from sklearn.preprocessing import StandardScaler
def preprocess_data(data):
    scaler = StandardScaler()
    pdb.set_trace() # Start debugger here
    scaled_data = scaler.fit_transform(data)
    normalized_data = scaled_data / np.linalg.norm(scaled_data)
    return normalized_data
# Example Data
data = np.array([[1, 2, 3], [4, 5, np.nan], [7, 8, 9]])
processed_data = preprocess_data(data)
print("Processed Data:\n", processed_data)