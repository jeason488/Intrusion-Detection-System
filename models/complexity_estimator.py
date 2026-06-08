import torch
import torch.nn as nn
import torch.nn.functional as F
class ComplexityEstimator(nn.Module):
    def __init__(self, config):
        super(ComplexityEstimator, self).__init__()
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.dataset_name = config.get('dataset_name', 'CICIDS2017')
        self.input_dim = config.get('feature_dim', 59)
        self.modal_indices = config['modal_indices']
        self.modal_dims = config['modal_dims']
        self.modal_types = config['modal_types']
        self.epsilon = float(config.get('complexity_epsilon', 1e-8))
        self.clip_value = float(config.get('complexity_clip_value', 10.0))
        self.modal_complexity_networks = nn.ModuleDict()
        n_modals = len(self.modal_types)
        self.weight_learning_network = nn.Sequential(
            nn.Linear(n_modals, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_modals),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.register_buffer(
            'running_weights',
            torch.ones(n_modals, device=self.device) / n_modals
        )
        self.weight_momentum = float(config.get('weight_momentum', 0.7))
        self.use_entropy_enhancement = config.get('use_entropy_enhancement', True)
        self.complexity_fusion_weights = {
            'intra': 0.25,
            'inter': 0.50,
            'qs_sim': 0.25,
            'network': 0.50
        }
    def _split_modalities(self, data):
        modalities = {}
        for modal_name, indices in self.modal_indices.items():
            if len(indices) > 0:
                modalities[modal_name] = data[:, indices].contiguous()
            else:
                modalities[modal_name] = torch.zeros(
                    data.size(0), 0,
                    device=data.device,
                    dtype=data.dtype
                )
        return modalities
    def _build_complexity_network(self, modal_dim):
        if modal_dim == 0:
            return None
        input_dim = 4 * modal_dim
        network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)
        return network
    def _ensure_numerical_stability(self, data):
        if data.numel() == 0:
            return data
        data = data.clone()
        data_mean = data.mean(dim=0, keepdim=True)
        data_std = data.std(dim=0, keepdim=True, unbiased=False) + self.epsilon
        data = (data - data_mean) / data_std
        data = torch.clamp(data, min=-self.clip_value, max=self.clip_value)
        data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
        data = torch.where(
            torch.isinf(data),
            torch.sign(data) * self.clip_value,
            data
        )
        return data
    def _compute_modal_statistics(self, support_modal, query_modal):
        if support_modal.size(1) == 0 or query_modal.size(1) == 0:
            return torch.zeros(1, device=self.device, requires_grad=True)
        support_modal = self._ensure_numerical_stability(support_modal)
        query_modal = self._ensure_numerical_stability(query_modal)
        combined_data = torch.cat([support_modal, query_modal], dim=0)
        mean_vals = combined_data.mean(dim=0)
        std_vals = combined_data.std(dim=0, unbiased=False) + self.epsilon
        min_vals = combined_data.min(dim=0)[0]
        max_vals = combined_data.max(dim=0)[0]
        stats = torch.cat([mean_vals, std_vals, min_vals, max_vals])
        return stats
    def _compute_intra_class_complexity(self, features, labels):
        if len(features) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        unique_labels = torch.unique(labels)
        n_classes = len(unique_labels)
        if n_classes <= 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        class_variances = []
        class_counts = []
        for label in unique_labels:
            mask = (labels == label)
            class_features = features[mask]
            if len(class_features) > 1:
                var = torch.var(class_features, dim=0, unbiased=False).mean()
                class_variances.append(var)
                class_counts.append(float(len(class_features)))
        if len(class_variances) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        class_variances = torch.stack(class_variances)
        class_counts = torch.tensor(class_counts, device=self.device)
        class_weights = class_counts / (class_counts.sum() + self.epsilon)
        weighted_var = (class_variances * class_weights).sum()
        normalized_complexity = torch.tanh(weighted_var)
        return normalized_complexity
    def _compute_inter_class_complexity(self, features, labels):
        if len(features) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        unique_labels = torch.unique(labels)
        n_classes = len(unique_labels)
        if n_classes <= 1:
            return torch.tensor(1.0, device=self.device, requires_grad=True)
        class_centers = []
        for label in unique_labels:
            mask = (labels == label)
            class_features = features[mask]
            center = class_features.mean(dim=0)
            class_centers.append(center)
        class_centers = torch.stack(class_centers)
        distances = torch.cdist(class_centers, class_centers, p=2)
        mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
        inter_distances = distances[mask]
        if len(inter_distances) == 0:
            return torch.tensor(0.5, device=self.device, requires_grad=True)
        mean_distance = inter_distances.mean()
        complexity = 1.0 / (mean_distance + self.epsilon)
        normalized_complexity = torch.sigmoid(complexity - 1.0)
        return normalized_complexity
    def _compute_query_support_similarity(self, support_features, query_features):
        if len(support_features) == 0 or len(query_features) == 0:
            return torch.tensor(0.5, device=self.device, requires_grad=True)
        support_mean = support_features.mean(dim=0)
        query_mean = query_features.mean(dim=0)
        similarity = F.cosine_similarity(
            support_mean.unsqueeze(0),
            query_mean.unsqueeze(0),
            dim=1
        ).squeeze()
        normalized_similarity = (similarity + 1.0) / 2.0
        return normalized_similarity
    def _compute_enhanced_modal_complexities(
            self,
            support_modalities,
            support_labels,
            query_modalities
    ):
        modal_complexities = {}
        for modal_name in self.modal_types:
            if modal_name not in support_modalities or modal_name not in query_modalities:
                modal_complexities[modal_name] = torch.tensor(
                    0.5, device=self.device, requires_grad=True
                )
                continue
            support_modal = support_modalities[modal_name]
            query_modal = query_modalities[modal_name]
            if support_modal.size(1) == 0:
                modal_complexities[modal_name] = torch.tensor(
                    0.5, device=self.device, requires_grad=True
                )
                continue
            intra_complexity = self._compute_intra_class_complexity(
                support_modal, support_labels
            )
            inter_complexity = self._compute_inter_class_complexity(
                support_modal, support_labels
            )
            qs_similarity = self._compute_query_support_similarity(
                support_modal, query_modal
            )
            stats_vec = self._compute_modal_statistics(support_modal, query_modal)
            if modal_name not in self.modal_complexity_networks:
                modal_dim = support_modal.size(1)
                self.modal_complexity_networks[modal_name] = \
                    self._build_complexity_network(modal_dim)
            if self.modal_complexity_networks[modal_name] is not None:
                stats_input = stats_vec.unsqueeze(0)
                network_complexity = self.modal_complexity_networks[modal_name](
                    stats_input
                ).squeeze()
            else:
                network_complexity = torch.tensor(
                    0.5, device=self.device, requires_grad=True
                )
            w = self.complexity_fusion_weights
            base_complexity = (
                    w['intra'] * intra_complexity +
                    w['inter'] * inter_complexity +
                    w['qs_sim'] * (1.0 - qs_similarity)
            )
            final_complexity = (
                    (1.0 - w['network']) * base_complexity +
                    w['network'] * network_complexity
            )
            modal_complexities[modal_name] = final_complexity
        return modal_complexities
    def _apply_entropy_enhancement(self, base_weights, complexities):
        if not self.use_entropy_enhancement:
            return base_weights
        entropy = -torch.sum(base_weights * torch.log(base_weights + self.epsilon))
        max_entropy = torch.log(
            torch.tensor(len(base_weights), dtype=torch.float32, device=self.device)
        )
        entropy_ratio = entropy / (max_entropy + self.epsilon)
        if entropy_ratio > 0.85:
            adaptive_weights = base_weights
        elif entropy_ratio < 0.5:
            temperature = 2.0
            adaptive_weights = F.softmax(complexities * temperature, dim=-1)
        else:
            alpha = (entropy_ratio - 0.5) / 0.35
            temperature = 1.5
            sharpened_weights = F.softmax(complexities * temperature, dim=-1)
            adaptive_weights = alpha * base_weights + (1.0 - alpha) * sharpened_weights
        return adaptive_weights
    def forward(self, support_data, support_labels, query_data):
        support_data = support_data.to(self.device)
        support_labels = support_labels.to(self.device)
        query_data = query_data.to(self.device)
        support_modalities = self._split_modalities(support_data)
        query_modalities = self._split_modalities(query_data)
        modal_complexities = self._compute_enhanced_modal_complexities(
            support_modalities, support_labels, query_modalities
        )
        complexity_values = torch.stack([
            modal_complexities.get(
                modal,
                torch.tensor(0.5, device=self.device, requires_grad=True)
            )
            for modal in self.modal_types
        ])
        base_weights = self.weight_learning_network(complexity_values)
        adaptive_weights = self._apply_entropy_enhancement(
            base_weights, complexity_values
        )
        with torch.no_grad():
            self.running_weights = (
                    self.weight_momentum * self.running_weights +
                    (1.0 - self.weight_momentum) * adaptive_weights
            )
        n_support = support_data.size(0)
        n_query = query_data.size(0)
        n_total = n_support + n_query
        modal_complexity_matrix = complexity_values.unsqueeze(0).expand(n_total, -1)
        global_complexity = torch.sum(
            modal_complexity_matrix * adaptive_weights.unsqueeze(0), dim=1
        )
        return {
            'modal_complexity': modal_complexity_matrix,
            'global_complexity': global_complexity,
            'modal_weights': adaptive_weights,
            'complexity_dict': {
                k: v.item() if v.numel() == 1 else v.mean().item()
                for k, v in modal_complexities.items()
            }
        }
    def estimate_complexity(self, support_data, support_labels, query_data):
        return self.forward(support_data, support_labels, query_data)
