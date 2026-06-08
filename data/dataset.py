import torch
from torch.utils.data import Dataset
class IDSDataset(Dataset):
    def __init__(self, data, labels, domain="source", window_idx=None, feature_columns=None):
        self.data = data
        self.labels = labels
        self.domain = domain
        self.window_idx = window_idx
        self.feature_columns = feature_columns
        self.attack_ratio = 0.0
        self.reused_samples = False
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
