class FeatureAligner:
    def __init__(self, common_features: list, feature_mapping: dict):
        self.common_features = common_features
        self.feature_mapping = feature_mapping
    def align(self, df, dataset_name: str) -> dict:
        aligned_features = {}
        dataset_mapping = self.feature_mapping.get(dataset_name, {})
        for feature in self.common_features:
            mapped_feature = dataset_mapping.get(feature, feature)
            if mapped_feature in df.columns:
                aligned_features[feature] = df[mapped_feature]
            else:
                if feature in ['duration', 'src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes', 'trans_depth', 'response_body_len']:
                    aligned_features[feature] = 0
                else:
                    aligned_features[feature] = 'unknown'
        return aligned_features
