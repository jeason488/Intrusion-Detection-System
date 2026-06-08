import pandas as pd
class LabelMapper:
    def __init__(self, label_mapping: dict, binary_classification: bool):
        self.label_mapping = label_mapping
        self.binary_classification = binary_classification
    def map(self, labels, dataset_name: str) -> pd.Series:
        dataset_mapping = self.label_mapping.get(dataset_name, {})
        mapped_labels = labels.map(dataset_mapping)
        mapped_labels = mapped_labels.fillna(1)
        return mapped_labels
