import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
class Normalizer:
    def __init__(self, normalization: str):
        self.normalization = normalization
        self.scaler = None
    def fit(self, data):
        if self.normalization == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.normalization == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
        self.scaler.fit(data)
    def transform(self, data, domain: str) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("Scaler not fitted")
        return self.scaler.transform(data)
