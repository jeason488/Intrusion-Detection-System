class DataConfig:
    def __init__(
        self,
        source_dataset: str,
        target_dataset: str,
        data_dir: str,
        common_features: list,
        feature_mapping: dict,
        label_mapping: dict,
        window_size: int,
        stride: int,
        window_mode: str,
        target_stream_mode: str,
        attack_ratio_schedule: list,
        ratio_min: float,
        ratio_max: float,
        shuffle_within_window: bool,
        binary_classification: bool,
        normalization: str,
        seed: int
    ):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.data_dir = data_dir
        self.common_features = common_features
        self.feature_mapping = feature_mapping
        self.label_mapping = label_mapping
        self.window_size = window_size
        self.stride = stride
        self.window_mode = window_mode
        self.target_stream_mode = target_stream_mode
        self.attack_ratio_schedule = attack_ratio_schedule
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.shuffle_within_window = shuffle_within_window
        self.binary_classification = binary_classification
        self.normalization = normalization
        self.seed = seed
