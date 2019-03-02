import numpy as np
from FeatureExtraction import VECTORIZED_DATA


# load dataset
print("Reading vectorized data...")
data = np.load(VECTORIZED_DATA)

print(data)