from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from data.config import DataConfig
from data.feature_aligner import FeatureAligner
from data.label_mapper import LabelMapper
from data.normalizer import Normalizer
from data.window_splitter import WindowSplitter
from data.dataset import IDSDataset
class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.data_config = self._create_data_config(config)
        self.feature_aligner = FeatureAligner(
            common_features=self.data_config.common_features,
            feature_mapping=self.data_config.feature_mapping
        )
        self.label_mapper = LabelMapper(
            label_mapping=self.data_config.label_mapping,
            binary_classification=self.data_config.binary_classification
        )
        self.normalizer = Normalizer(normalization=self.data_config.normalization)
        self.window_splitter = WindowSplitter(
            window_size=self.data_config.window_size,
            stride=self.data_config.stride,
            window_mode=self.data_config.window_mode
        )
        self.category_maps = {}
        self.category_sizes = {}
    def _create_data_config(self, config) -> DataConfig:
        common_features = [
            'duration',
            'proto',
            'service',
            'state',
            'src_pkts',
            'dst_pkts',
            'src_bytes',
            'dst_bytes',
            'trans_depth',
            'response_body_len'
        ]
        feature_mapping = {
            'UNSW-NB15': {
                'duration': 'dur',
                'proto': 'proto',
                'service': 'service',
                'state': 'state',
                'src_pkts': 'spkts',
                'dst_pkts': 'dpkts',
                'src_bytes': 'sbytes',
                'dst_bytes': 'dbytes',
                'trans_depth': 'trans_depth',
                'response_body_len': 'response_body_len'
            },
            'TON-IoT': {
                'duration': 'duration',
                'proto': 'proto',
                'service': 'service',
                'state': 'conn_state',
                'src_pkts': 'src_pkts',
                'dst_pkts': 'dst_pkts',
                'src_bytes': 'src_bytes',
                'dst_bytes': 'dst_bytes',
                'trans_depth': 'http_trans_depth',
                'response_body_len': 'http_response_body_len'
            },
            'CICIDS2017': {
                'duration': 'Duration',
                'proto': 'Protocol',
                'service': 'Service',
                'state': 'State',
                'src_pkts': 'SrcPackets',
                'dst_pkts': 'DstPackets',
                'src_bytes': 'SrcBytes',
                'dst_bytes': 'DstBytes',
                'trans_depth': 'TransDepth',
                'response_body_len': 'ResponseBodyLength'
            }
        }
        label_mapping = {
            'UNSW-NB15': {
                'normal': 0,
                'benign': 0,
                '0': 0,
                'attack': 1,
                'malicious': 1,
                '1': 1
            },
            'TON-IoT': {
                'normal': 0,
                'benign': 0,
                '0': 0,
                'attack': 1,
                'malicious': 1,
                '1': 1
            },
            'CICIDS2017': {
                'BENIGN': 0,
                '0': 0,
                'attack': 1,
                'malicious': 1,
                '1': 1
            }
        }
        target_stream_mode = config.get('drift', {}).get('mode', 'natural')
        attack_ratio_schedule = config.get('drift', {}).get('attack_ratio_schedule', [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3])
        ratio_min = config.get('drift', {}).get('ratio_min', 0.3)
        ratio_max = config.get('drift', {}).get('ratio_max', 0.7)
        shuffle_within_window = config.get('drift', {}).get('shuffle_within_window', False)
        return DataConfig(
            source_dataset=config['data'].get('source_dataset', 'UNSW-NB15'),
            target_dataset=config['data'].get('target_dataset', 'TON-IoT'),
            data_dir='./data',
            common_features=common_features,
            feature_mapping=feature_mapping,
            label_mapping=label_mapping,
            window_size=config['data'].get('window_size', 100),
            stride=config['data'].get('window_size', 100) // 2,
            window_mode=config['data'].get('window_mode', 'sliding'),
            target_stream_mode=target_stream_mode,
            attack_ratio_schedule=attack_ratio_schedule,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            shuffle_within_window=shuffle_within_window,
            binary_classification=True,
            normalization=config['data'].get('normalization', 'minmax'),
            seed=config.get('seed', 42)
        )
    def get_source_dataloader(self, batch_size: int = None, split_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        from sklearn.model_selection import StratifiedShuffleSplit
        if batch_size is None:
            batch_size = self.config['data'].get('batch_size', 32)
        source_df = self._load_dataset(self.data_config.source_dataset)
        print(f"Source dataset shape: {source_df.shape}")
        print(f"Source dataset label distribution: {source_df['label'].value_counts()}")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-split_ratio, random_state=self.data_config.seed)
        labels = self.label_mapper.map(source_df['label'], self.data_config.source_dataset)
        for train_idx, val_idx in sss.split(source_df, labels):
            train_df = source_df.iloc[train_idx]
            val_df = source_df.iloc[val_idx]
        print(f"Train set size: {len(train_df)}, Val set size: {len(val_df)}")
        train_labels = self.label_mapper.map(train_df['label'], self.data_config.source_dataset)
        val_labels = self.label_mapper.map(val_df['label'], self.data_config.source_dataset)
        print(f"Train set label distribution: {train_labels.value_counts()}")
        print(f"Val set label distribution: {val_labels.value_counts()}")
        train_dataset = self.process_source(train_df, 'label', fit_normalizer=True)
        val_dataset = self.process_source(val_df, 'label', fit_normalizer=False)
        print("normalizer fitted on train only: True")
        print("val transformed using train normalizer: True")
        print("target transformed using train normalizer: True")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        return train_loader, val_loader
    def get_target_windows(self) -> List[Tuple[np.ndarray, np.ndarray, str, float, bool]]:
        target_df = self._load_dataset(self.data_config.target_dataset)
        print(f"Target dataset shape: {target_df.shape}")
        print(f"Target dataset label distribution: {target_df['label'].value_counts()}")
        target_datasets = self.process_target(target_df, 'label', None)
        windows = []
        for i, dataset in enumerate(target_datasets):
            features = dataset.data
            labels = dataset.labels
            attack_ratio = getattr(dataset, 'attack_ratio', 0.5)
            reused_samples = getattr(dataset, 'reused_samples', False)
            if i == 0:
                drift_type = 'initial'
            else:
                if self.data_config.target_stream_mode == 'natural':
                    drift_type = 'natural_temporal_stream'
                elif self.data_config.target_stream_mode == 'smooth_ratio_shift':
                    drift_type = 'ratio_shift'
                else:
                    drift_type = 'unknown'
            windows.append((features, labels, drift_type, attack_ratio, reused_samples))
        return windows
    def _load_dataset(self, dataset_name: str) -> pd.DataFrame:
        file_map = {
            'UNSW-NB15': 'data/unsw_nb15.csv',
            'TON-IoT': 'data/ton_iot.csv',
            'CICIDS2017': 'data/cicids2017.csv'
        }
        file_path = file_map.get(dataset_name)
        if not file_path:
            raise ValueError(f"Dataset {dataset_name} not supported")
        print(f"Loading {dataset_name} dataset from {file_path}...")
        return pd.read_csv(file_path)
    def process_source(self, source_df: pd.DataFrame, label_column: str, fit_normalizer: bool = False) -> IDSDataset:
        aligned_df = self.feature_aligner.align(source_df, self.data_config.source_dataset)
        enhanced_df = self._enhance_numeric_features(aligned_df)
        labels = self.label_mapper.map(source_df[label_column], self.data_config.source_dataset)
        encoded_df = self._encode_categorical_features(enhanced_df, fit=fit_normalizer)
        numerical_cols = ['duration', 'src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes', 'trans_depth', 'response_body_len', 'log_src_bytes', 'log_dst_bytes', 'bytes_ratio', 'pkts_ratio', 'bytes_per_pkt']
        categorical_cols = ['proto', 'service', 'state']
        ordered_columns = numerical_cols + categorical_cols
        ordered_columns = [col for col in ordered_columns if col in encoded_df.columns]
        if hasattr(self, 'feature_columns') and not fit_normalizer:
            encoded_df = encoded_df[self.feature_columns]
        elif fit_normalizer:
            self.feature_columns = ordered_columns
            encoded_df = encoded_df[ordered_columns]
        numerical_cols = ['duration', 'src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes', 'trans_depth', 'response_body_len', 'log_src_bytes', 'log_dst_bytes', 'bytes_ratio', 'pkts_ratio', 'bytes_per_pkt']
        categorical_cols = ['proto', 'service', 'state']
        numerical_df = encoded_df[numerical_cols]
        categorical_df = encoded_df[categorical_cols]
        if fit_normalizer:
            self.normalizer.fit(numerical_df)
        normalized_numerical = self.normalizer.transform(numerical_df, "source")
        normalized_data = np.hstack([normalized_numerical, categorical_df.values])
        return IDSDataset(
            data=normalized_data,
            labels=labels.values,
            domain="source",
            feature_columns=self.feature_columns
        )
    def process_target(self, target_df: pd.DataFrame, label_column: str, time_column: str) -> List[IDSDataset]:
        aligned_df = self.feature_aligner.align(target_df, self.data_config.target_dataset)
        enhanced_df = self._enhance_numeric_features(aligned_df)
        labels = self.label_mapper.map(target_df[label_column], self.data_config.target_dataset)
        encoded_df = self._encode_categorical_features(enhanced_df, fit=False)
        numerical_cols = ['duration', 'src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes', 'trans_depth', 'response_body_len', 'log_src_bytes', 'log_dst_bytes', 'bytes_ratio', 'pkts_ratio', 'bytes_per_pkt']
        categorical_cols = ['proto', 'service', 'state']
        numerical_df = encoded_df[numerical_cols]
        categorical_df = encoded_df[categorical_cols]
        normalized_numerical = self.normalizer.transform(numerical_df, "target")
        normalized_data = np.hstack([normalized_numerical, categorical_df.values])
        window_data = []
        windows_with_positions = []
        target_stream_mode = self.data_config.target_stream_mode
        print(f"[Target Stream] mode={target_stream_mode}")
        if target_stream_mode == "natural":
            total_samples = len(target_df)
            window_size = self.data_config.window_size
            stride = self.data_config.stride
            num_windows = max(0, (total_samples - window_size) // stride + 1)
            print(f"Calculated window count: {num_windows}")
            window_idx = 0
            start = 0
            while start + window_size <= total_samples:
                end = start + window_size
                window_df = target_df.iloc[start:end]
                window_labels = labels.iloc[start:end]
                attack_ratio = window_labels.mean()
                windows_with_positions.append((window_idx, window_df, start, end, attack_ratio, False))
                window_idx += 1
                start += stride
        elif target_stream_mode == "smooth_ratio_shift":
            total_samples = len(target_df)
            window_size = self.data_config.window_size
            stride = self.data_config.stride
            num_windows = max(0, (total_samples - window_size) // stride + 1)
            print(f"Calculated window count: {num_windows}")
            normal_indices = list(target_df[labels == 0].index)
            attack_indices = list(target_df[labels == 1].index)
            print(f"Normal samples: {len(normal_indices)}, Attack samples: {len(attack_indices)}")
            attack_ratio_schedule = self.data_config.attack_ratio_schedule
            if attack_ratio_schedule is None:
                attack_ratio_schedule = [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3]
            attack_ratios = []
            for i in range(num_windows):
                if i < len(attack_ratio_schedule):
                    attack_ratios.append(attack_ratio_schedule[i])
                else:
                    attack_ratios.append(attack_ratio_schedule[i % len(attack_ratio_schedule)])
            normal_ptr = 0
            attack_ptr = 0
            window_idx = 0
            while window_idx < num_windows:
                scheduled_ratio = attack_ratios[window_idx]
                scheduled_ratio = max(self.data_config.ratio_min, min(self.data_config.ratio_max, scheduled_ratio))
                attack_count = int(window_size * scheduled_ratio)
                normal_count = window_size - attack_count
                if len(normal_indices) < normal_count or len(attack_indices) < attack_count:
                    break
                selected_normal = []
                reused_normal = False
                for i in range(normal_count):
                    if normal_ptr >= len(normal_indices):
                        normal_ptr = 0
                        reused_normal = True
                    selected_normal.append(normal_indices[normal_ptr])
                    normal_ptr += 1
                selected_attack = []
                reused_attack = False
                for i in range(attack_count):
                    if attack_ptr >= len(attack_indices):
                        attack_ptr = 0
                        reused_attack = True
                    selected_attack.append(attack_indices[attack_ptr])
                    attack_ptr += 1
                selected_indices = sorted(selected_normal + selected_attack)
                if len(selected_indices) == window_size:
                    window_df = target_df.iloc[selected_indices]
                    window_labels = labels.iloc[selected_indices]
                    actual_ratio = window_labels.mean()
                    reused_samples = reused_normal or reused_attack
                    windows_with_positions.append((window_idx, window_df, 0, 0, actual_ratio, reused_samples, scheduled_ratio))
                    window_idx += 1
        else:
            total_samples = len(target_df)
            window_size = self.data_config.window_size
            stride = self.data_config.stride
            num_windows = max(0, (total_samples - window_size) // stride + 1)
            print(f"Calculated window count: {num_windows}")
            window_idx = 0
            start = 0
            while start + window_size <= total_samples:
                end = start + window_size
                window_df = target_df.iloc[start:end]
                window_labels = labels.iloc[start:end]
                attack_ratio = window_labels.mean()
                windows_with_positions.append((window_idx, window_df, start, end, attack_ratio, False))
                window_idx += 1
                start += stride
        for item in windows_with_positions:
            if len(item) == 6:
                window_idx, window_df, start, end, attack_ratio, reused_samples = item
                scheduled_ratio = None
            else:
                window_idx, window_df, _, _, actual_ratio, reused_samples, scheduled_ratio = item
                attack_ratio = actual_ratio
                start = window_df.index[0]
                end = window_df.index[-1] + 1
            if target_stream_mode == "natural":
                print(f"Window {window_idx}: range={start}->{end}, size={len(window_df)}, attack_ratio={attack_ratio:.4f}, actual_drift_type={'initial' if window_idx == 0 else 'natural_temporal_stream'}")
            else:
                print(f"Window {window_idx}: scheduled_ratio={scheduled_ratio:.2f}, actual_ratio={attack_ratio:.4f}, size={len(window_df)}, reused_samples={reused_samples}, actual_drift_type={'initial' if window_idx == 0 else 'ratio_shift'}")
            window_indices = window_df.index
            window_normalized_data = normalized_data[window_indices]
            window_labels = labels.iloc[window_indices].values
            window_dataset = IDSDataset(
                data=window_normalized_data,
                labels=window_labels,
                domain="target",
                window_idx=window_idx,
                feature_columns=self.feature_columns
            )
            window_dataset.attack_ratio = attack_ratio
            window_dataset.reused_samples = reused_samples
            window_data.append(window_dataset)
        return window_data
    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        encoded_df = df.copy()
        categorical_features = ['proto', 'service', 'state']
        for feature in categorical_features:
            if feature in encoded_df.columns:
                if fit:
                    unique_values = encoded_df[feature].unique()
                    value_map = {v: i+1 for i, v in enumerate(unique_values)}
                    self.category_maps[feature] = value_map
                    self.category_sizes[feature] = len(value_map) + 1
                    encoded_df[feature] = encoded_df[feature].map(value_map).fillna(0)
                else:
                    if feature in self.category_maps:
                        value_map = self.category_maps[feature]
                        encoded_df[feature] = encoded_df[feature].map(value_map).fillna(0)
                encoded_df[feature] = encoded_df[feature].astype(int)
                encoded_df[feature] = encoded_df[feature].clip(lower=0)
        for col in encoded_df.columns:
            if col not in ['proto', 'service', 'state']:
                encoded_df[col] = pd.to_numeric(encoded_df[col], errors="coerce").fillna(0.0)
        return encoded_df
    def _enhance_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        enhanced_df = df.copy()
        if 'src_bytes' in enhanced_df.columns:
            enhanced_df['log_src_bytes'] = np.log1p(enhanced_df['src_bytes'])
        if 'dst_bytes' in enhanced_df.columns:
            enhanced_df['log_dst_bytes'] = np.log1p(enhanced_df['dst_bytes'])
        if 'src_bytes' in enhanced_df.columns and 'dst_bytes' in enhanced_df.columns:
            enhanced_df['bytes_ratio'] = enhanced_df['src_bytes'] / (enhanced_df['dst_bytes'] + 1)
        if 'src_pkts' in enhanced_df.columns and 'dst_pkts' in enhanced_df.columns:
            enhanced_df['pkts_ratio'] = enhanced_df['src_pkts'] / (enhanced_df['dst_pkts'] + 1)
        if all(col in enhanced_df.columns for col in ['src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts']):
            enhanced_df['bytes_per_pkt'] = (enhanced_df['src_bytes'] + enhanced_df['dst_bytes']) / (enhanced_df['src_pkts'] + enhanced_df['dst_pkts'] + 1)
        return enhanced_df
    def get_feature_dim(self) -> int:
        if hasattr(self, 'feature_columns'):
            return len(self.feature_columns)
        return self.config['data'].get('input_dim', 78)
    def get_num_classes(self) -> int:
        return 2 if self.data_config.binary_classification else len(set(
            label for dataset_labels in self.data_config.label_mapping.values()
            for label in dataset_labels.values()
        ))
