import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, List, Optional, Any
import logging
logger = logging.getLogger(__name__)
class HierarchyConstructor(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(HierarchyConstructor, self).__init__()
        self.device = config.get(
            'device',
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.modal_types = config.get('modal_types', ['numerical'])
        self.n_modalities = len(self.modal_types)
        self.prototype_dim = config.get('prototype_dim', 128)
        self.max_prototypes_per_class = config.get('max_prototypes_per_class', 5)
        self.min_prototypes_for_minority = config.get('min_prototypes_for_minority', 2)
        self.hierarchy_epsilon = float(config.get('hierarchy_epsilon', 1e-8))
        self.prototype_main_strategy = config.get('prototype_main_strategy', 'complexity_weighted')
        self.prototype_main_alpha = float(config.get('prototype_main_alpha', 0.3))
        self.prototype_temperature = float(config.get('prototype_temperature', 0.5))
        self.high_complexity_threshold = config.get('high_complexity_threshold', 0.7)
        self.low_complexity_threshold = config.get('low_complexity_threshold', 0.3)
        self.min_norm_threshold = float(config.get('min_norm_threshold', 1e-6))
        self.clip_value = float(config.get('hierarchy_clip_value', 10.0))
        self.sub_prototype_generator = nn.Sequential(
            nn.Linear(self.prototype_dim + self.n_modalities, self.prototype_dim),
            nn.LayerNorm(self.prototype_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(self.device)
        self.complexity_weight_generator = nn.Sequential(
            nn.Linear(self.prototype_dim + self.n_modalities, 1),
            nn.Sigmoid()
        ).to(self.device)
        confidence_input_dim = self.prototype_dim + self.n_modalities + 1
        self.confidence_estimator = nn.Sequential(
            nn.Linear(confidence_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)
        self._total_subclusters_created = 0
        self._hierarchy_build_count = 0
    def build_hierarchy(
            self,
            support_features: torch.Tensor,
            complexity_info: Dict[str, Any],
            labels: torch.Tensor,
            class_stats: Optional[Dict[int, Dict[str, Any]]] = None,
            accuracy_feedback_per_class: Optional[Dict[int, float]] = None
    ) -> Dict[int, Dict[str, Any]]:
        self._hierarchy_build_count += 1
        modal_complexity = complexity_info.get('modal_complexity')
        global_complexity = complexity_info.get('global_complexity')
        n_support = len(support_features)
        if modal_complexity is not None:
            if modal_complexity.size(0) >= n_support:
                modal_complexity_support = modal_complexity[:n_support]
            else:
                padding = torch.zeros(
                    n_support - modal_complexity.size(0),
                    self.n_modalities,
                    device=self.device
                )
                modal_complexity_support = torch.cat([modal_complexity, padding], dim=0)
        else:
            modal_complexity_support = torch.zeros(
                n_support, self.n_modalities, device=self.device
            )
        if isinstance(global_complexity, torch.Tensor):
            if global_complexity.numel() == 0:
                global_complexity_scalar = 0.5
                global_complexity_support = torch.full(
                    (n_support,), 0.5, device=self.device
                )
            elif global_complexity.numel() > 1:
                if global_complexity.size(0) >= n_support:
                    global_complexity_support = global_complexity[:n_support]
                else:
                    padding = torch.full(
                        (n_support - global_complexity.size(0),),
                        0.5,
                        device=self.device
                    )
                    global_complexity_support = torch.cat([global_complexity, padding], dim=0)
                global_complexity_scalar = global_complexity_support.mean().item()
            else:
                global_complexity_scalar = global_complexity.item()
                global_complexity_support = torch.full(
                    (n_support,), global_complexity_scalar, device=self.device
                )
        else:
            global_complexity_scalar = float(global_complexity) if global_complexity is not None else 0.5
            global_complexity_support = torch.full(
                (n_support,), global_complexity_scalar, device=self.device
            )
        unique_labels = torch.unique(labels)
        hierarchical_prototypes = {}
        for class_id in unique_labels:
            class_id_item = class_id.item()
            class_mask = (labels == class_id)
            class_features = support_features[class_mask]
            n_samples = len(class_features)
            if n_samples == 0:
                continue
            class_complexity_data = {
                'modal_complexity': modal_complexity_support[class_mask],
                'global_complexity': global_complexity_support[class_mask]
            }
            class_complexity = class_complexity_data['global_complexity'].mean().item()
            feedback_acc = accuracy_feedback_per_class.get(
                class_id_item, None
            ) if accuracy_feedback_per_class else None
            main_prototype_feature = self._build_enhanced_main_prototype_v2(
                class_features, class_complexity_data
            )
            main_prototype = {
                'feature': main_prototype_feature,
                'weight': 1.0,
                'n_samples': n_samples
            }
            n_sub_prototypes = self._determine_enhanced_sub_prototype_count(
                class_features, class_complexity_data, class_stats, feedback_acc
            )
            sub_prototypes = []
            if n_sub_prototypes > 0:
                sub_prototypes = self._build_enhanced_sub_prototypes(
                    class_features,
                    class_complexity_data,
                    main_prototype_feature,
                    n_sub_prototypes
                )
            hierarchical_prototypes[class_id_item] = {
                'main': main_prototype,
                'sub': sub_prototypes,
                'complexity': class_complexity,
                'n_subclusters': len(sub_prototypes),
                'class_complexity_info': {
                    'mean_complexity': class_complexity,
                    'std_complexity': class_complexity_data['global_complexity'].std(unbiased=False).item(),
                    'n_samples': n_samples,
                    'n_sub_prototypes': n_sub_prototypes
                }
            }
            if feedback_acc is not None:
                hierarchical_prototypes[class_id_item]['accuracy'] = feedback_acc
        return hierarchical_prototypes
    def _build_enhanced_main_prototype_v2(
            self,
            class_features: torch.Tensor,
            class_complexity_data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if class_features.numel() == 0:
            return torch.randn(self.prototype_dim, device=self.device) * 0.1
        input_norm = torch.norm(class_features).item()
        if input_norm < self.min_norm_threshold:
            class_features = class_features + torch.randn_like(class_features) * 0.01
        if self.prototype_main_strategy == 'feature_mean':
            main_proto = class_features.mean(dim=0)
        elif self.prototype_main_strategy == 'complexity_weighted':
            weights = class_complexity_data['global_complexity']
            weights_sum = weights.sum() + self.hierarchy_epsilon
            weights = weights / weights_sum
            main_proto = (class_features * weights.unsqueeze(-1)).sum(dim=0)
        else:
            mean_proto = class_features.mean(dim=0)
            weights = class_complexity_data['global_complexity']
            weights_sum = weights.sum() + self.hierarchy_epsilon
            weights = weights / weights_sum
            weighted_proto = (class_features * weights.unsqueeze(-1)).sum(dim=0)
            main_proto = (
                    self.prototype_main_alpha * weighted_proto +
                    (1.0 - self.prototype_main_alpha) * mean_proto
            )
        output_norm = torch.norm(main_proto).item()
        if output_norm < self.min_norm_threshold:
            main_proto = class_features.mean(dim=0)
        main_proto = self._safe_normalize(main_proto)
        return main_proto
    def _determine_enhanced_sub_prototype_count(
            self,
            class_features: torch.Tensor,
            class_complexity_data: Dict[str, torch.Tensor],
            class_stats: Optional[Dict] = None,
            feedback_acc: Optional[float] = None
    ) -> int:
        complexity_mean = class_complexity_data['global_complexity'].mean().item()
        complexity_std = class_complexity_data['global_complexity'].std(unbiased=False).item()
        n_samples = len(class_features)
        if feedback_acc is not None:
            if feedback_acc > 0.9:
                base_count = self.min_prototypes_for_minority
            elif feedback_acc < 0.7:
                base_count = self.max_prototypes_per_class
            else:
                ratio = (0.9 - feedback_acc) / 0.2
                base_count = int(
                    self.min_prototypes_for_minority +
                    ratio * (self.max_prototypes_per_class - self.min_prototypes_for_minority)
                )
        elif complexity_mean > self.high_complexity_threshold and complexity_std > 0.15:
            base_count = self.max_prototypes_per_class
        elif complexity_mean < self.low_complexity_threshold:
            base_count = self.min_prototypes_for_minority
        else:
            ratio = (complexity_mean - self.low_complexity_threshold) / \
                    (self.high_complexity_threshold - self.low_complexity_threshold)
            base_count = int(
                self.min_prototypes_for_minority +
                ratio * (self.max_prototypes_per_class - self.min_prototypes_for_minority)
            )
        final_count = min(base_count, max(0, n_samples - 1))
        return final_count
    def _build_enhanced_sub_prototypes(
            self,
            class_features: torch.Tensor,
            class_complexity_data: Dict[str, torch.Tensor],
            main_prototype: torch.Tensor,
            n_sub_prototypes: int
    ) -> List[Dict[str, Any]]:
        if n_sub_prototypes == 0 or len(class_features) == 0:
            return []
        selected_indices = self._enhanced_select_representative_samples_v2(
            class_features, class_complexity_data, n_sub_prototypes
        )
        if not selected_indices:
            return []
        selected_features = class_features[selected_indices]
        sample_modal_complexities = class_complexity_data['modal_complexity'][selected_indices]
        sub_proto_inputs = torch.cat([selected_features, sample_modal_complexities], dim=1)
        sub_proto_features = self.sub_prototype_generator(sub_proto_inputs)
        sub_proto_features = self._safe_normalize(sub_proto_features)
        complexity_weight_inputs = torch.cat([sub_proto_features, sample_modal_complexities], dim=1)
        complexity_weights = self.complexity_weight_generator(complexity_weight_inputs).squeeze(-1)
        contrastive_confidences = self._compute_contrastive_confidence(
            sub_proto_features, class_features
        )
        confidence_input = torch.cat([
            sub_proto_features,
            sample_modal_complexities,
            complexity_weights.unsqueeze(1)
        ], dim=1)
        expected_dim = self.confidence_estimator[0].in_features
        actual_dim = confidence_input.size(1)
        if actual_dim != expected_dim:
            if actual_dim < expected_dim:
                padding = torch.zeros(
                    len(selected_indices),
                    expected_dim - actual_dim,
                    device=self.device
                )
                confidence_input = torch.cat([confidence_input, padding], dim=1)
            else:
                confidence_input = confidence_input[:, :expected_dim]
        network_confidences = self.confidence_estimator(confidence_input).squeeze(-1)
        final_confidences = 0.6 * contrastive_confidences + 0.4 * network_confidences
        variances = []
        for i, idx in enumerate(selected_indices):
            distances = torch.norm(
                class_features - sub_proto_features[i].unsqueeze(0), dim=1
            )
            k = min(5, len(class_features))
            nearest_features = class_features[distances.topk(k, largest=False).indices]
            variance = torch.var(nearest_features, dim=0, unbiased=False).mean().item()
            variances.append(variance)
        sub_prototypes = []
        n_total = len(selected_indices)
        for i, idx in enumerate(selected_indices):
            sub_proto = {
                'feature': sub_proto_features[i],
                'confidence': final_confidences[i].item(),
                'complexity_weight': complexity_weights[i].item(),
                'modal_type': 'mixed',
                'source_complexity': sample_modal_complexities[i].cpu().tolist(),
                'sample_index': idx,
                'weight': 1.0 / n_total,
                'n_samples': 1,
                'variance': variances[i]
            }
            sub_prototypes.append(sub_proto)
        self._total_subclusters_created += len(sub_prototypes)
        return sub_prototypes
    def _safe_normalize(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return features
        if features.dim() == 1:
            norm = torch.norm(features, p=2) + self.hierarchy_epsilon
            if norm < self.min_norm_threshold:
                normalized = F.normalize(
                    torch.randn_like(features), p=2, dim=0
                )
            else:
                normalized = features / norm
        else:
            norm = torch.norm(features, p=2, dim=1, keepdim=True) + self.hierarchy_epsilon
            zero_mask = (norm.squeeze() < self.min_norm_threshold)
            if zero_mask.any():
                features[zero_mask] = torch.randn_like(features[zero_mask]) * 0.1
                norm = torch.norm(features, p=2, dim=1, keepdim=True) + self.hierarchy_epsilon
            normalized = features / norm
        if torch.isnan(normalized).any() or torch.isinf(normalized).any():
            normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        return normalized
    def _enhanced_select_representative_samples_v2(
            self,
            class_features: torch.Tensor,
            class_complexity_data: Dict[str, torch.Tensor],
            n_select: int
    ) -> List[int]:
        n_samples = len(class_features)
        if n_samples == 0 or n_select == 0:
            return []
        if n_samples <= n_select:
            return list(range(n_samples))
        global_complexities = class_complexity_data['modal_complexity'].sum(dim=1)
        n_init = max(1, n_select // 2)
        init_indices = self._kmeans_plusplus_init(
            class_features, global_complexities, n_init
        )
        remaining = n_select - len(init_indices)
        if remaining > 0:
            diverse_indices = self._diversity_maximization(
                class_features, global_complexities, init_indices, remaining
            )
            selected_indices = init_indices + diverse_indices
        else:
            selected_indices = init_indices
        return selected_indices
    def _kmeans_plusplus_init(
            self,
            features: torch.Tensor,
            complexities: torch.Tensor,
            k: int
    ) -> List[int]:
        n = len(features)
        if n == 0 or k == 0:
            return []
        k = min(k, n)
        selected = []
        first_idx = complexities.argmax().item()
        selected.append(first_idx)
        for _ in range(k - 1):
            selected_features = features[selected]
            distances = torch.cdist(features, selected_features).min(dim=1)[0]
            probs = distances / (distances.sum() + self.hierarchy_epsilon)
            next_idx = torch.multinomial(probs, 1).item()
            if next_idx not in selected:
                selected.append(next_idx)
            else:
                sorted_indices = distances.argsort(descending=True)
                for idx in sorted_indices:
                    if idx.item() not in selected:
                        selected.append(idx.item())
                        break
        return selected
    def _diversity_maximization(
            self,
            features: torch.Tensor,
            complexities: torch.Tensor,
            init_indices: List[int],
            k: int
    ) -> List[int]:
        if k == 0 or len(features) == 0:
            return []
        n = len(features)
        selected_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        selected_mask[init_indices] = True
        selected = []
        for _ in range(k):
            candidate_mask = ~selected_mask
            if not candidate_mask.any():
                break
            candidate_features = features[candidate_mask]
            candidate_indices = torch.arange(n, device=self.device)[candidate_mask]
            selected_features = features[selected_mask]
            distances = torch.cdist(candidate_features, selected_features).min(dim=1)[0]
            best_candidate = distances.argmax()
            best_idx = candidate_indices[best_candidate].item()
            selected.append(best_idx)
            selected_mask[best_idx] = True
        return selected
    def _compute_contrastive_confidence(
            self,
            sub_proto_features: torch.Tensor,
            class_features: torch.Tensor
    ) -> torch.Tensor:
        if len(sub_proto_features) == 0 or len(class_features) == 0:
            return torch.ones(len(sub_proto_features), device=self.device) * 0.5
        intra_similarities = F.cosine_similarity(
            sub_proto_features.unsqueeze(1),
            class_features.unsqueeze(0),
            dim=2
        ).mean(dim=1)
        confidence = torch.sigmoid(intra_similarities)
        return confidence
    def adapt_prototypes_from_query(
            self,
            current_prototypes: Dict[int, Dict[str, Any]],
            query_features: torch.Tensor,
            query_predictions: torch.Tensor,
            confidence_threshold: float = 0.8
    ) -> Dict[int, Dict[str, Any]]:
        adapted_prototypes = {}
        for class_id, proto_dict in current_prototypes.items():
            class_predictions = (query_predictions.argmax(dim=1) == class_id)
            class_confidences = torch.softmax(query_predictions, dim=1)[:, class_id]
            high_confidence_mask = class_predictions & (class_confidences > confidence_threshold)
            if not high_confidence_mask.any():
                adapted_prototypes[class_id] = self._deep_copy_prototype(proto_dict)
                continue
            selected_query_features = query_features[high_confidence_mask]
            current_main = proto_dict['main']['feature']
            query_mean = selected_query_features.mean(dim=0)
            adaptation_rate = 0.1
            updated_main_feature = (
                    (1 - adaptation_rate) * current_main +
                    adaptation_rate * query_mean
            )
            updated_main_feature = self._safe_normalize(updated_main_feature)
            updated_main = {
                'feature': updated_main_feature,
                'weight': proto_dict['main']['weight'],
                'n_samples': proto_dict['main']['n_samples']
            }
            updated_sub_prototypes = []
            for sub_proto in proto_dict['sub']:
                sub_feature = sub_proto['feature']
                distances = torch.norm(
                    selected_query_features - sub_feature.unsqueeze(0), dim=1
                )
                k = min(3, len(selected_query_features))
                nearest_indices = distances.topk(k, largest=False).indices
                nearest_features = selected_query_features[nearest_indices]
                sub_adaptation_rate = 0.05
                updated_sub_feature = (
                        (1 - sub_adaptation_rate) * sub_feature +
                        sub_adaptation_rate * nearest_features.mean(dim=0)
                )
                updated_sub_feature = self._safe_normalize(updated_sub_feature)
                consistency = (distances[nearest_indices] < distances.median()).float().mean()
                updated_confidence = 0.8 * sub_proto['confidence'] + 0.2 * consistency.item()
                updated_sub_proto = {
                    'feature': updated_sub_feature,
                    'confidence': updated_confidence,
                    'complexity_weight': sub_proto['complexity_weight'],
                    'modal_type': sub_proto['modal_type'],
                    'source_complexity': sub_proto['source_complexity'],
                    'sample_index': sub_proto['sample_index'],
                    'weight': sub_proto['weight'],
                    'n_samples': sub_proto['n_samples'],
                    'variance': sub_proto['variance']
                }
                updated_sub_prototypes.append(updated_sub_proto)
            adapted_prototypes[class_id] = {
                'main': updated_main,
                'sub': updated_sub_prototypes,
                'complexity': proto_dict['complexity'],
                'n_subclusters': len(updated_sub_prototypes),
                'class_complexity_info': proto_dict['class_complexity_info'].copy()
            }
            if 'accuracy' in proto_dict:
                adapted_prototypes[class_id]['accuracy'] = proto_dict['accuracy']
        return adapted_prototypes
    def _deep_copy_prototype(self, proto_dict: Dict[str, Any]) -> Dict[str, Any]:
        copied_proto = {
            'main': {
                'feature': proto_dict['main']['feature'].clone(),
                'weight': proto_dict['main']['weight'],
                'n_samples': proto_dict['main']['n_samples']
            },
            'sub': [],
            'complexity': proto_dict['complexity'],
            'n_subclusters': proto_dict['n_subclusters'],
            'class_complexity_info': proto_dict['class_complexity_info'].copy()
        }
        for sub_proto in proto_dict['sub']:
            copied_sub = {
                'feature': sub_proto['feature'].clone(),
                'confidence': sub_proto['confidence'],
                'complexity_weight': sub_proto['complexity_weight'],
                'modal_type': sub_proto['modal_type'],
                'source_complexity': sub_proto['source_complexity'].copy(),
                'sample_index': sub_proto['sample_index'],
                'weight': sub_proto['weight'],
                'n_samples': sub_proto['n_samples'],
                'variance': sub_proto['variance']
            }
            copied_proto['sub'].append(copied_sub)
        if 'accuracy' in proto_dict:
            copied_proto['accuracy'] = proto_dict['accuracy']
        return copied_proto
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_subclusters_created': self._total_subclusters_created,
            'hierarchy_build_count': self._hierarchy_build_count,
            'device': str(self.device),
            'n_modalities': self.n_modalities,
            'prototype_dim': self.prototype_dim
        }
    def reset_statistics(self):
        self._total_subclusters_created = 0
        self._hierarchy_build_count = 0
