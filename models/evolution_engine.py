import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy
class PrototypeEvolutionEngine(nn.Module):
    def __init__(self, config: Dict):
        super(PrototypeEvolutionEngine, self).__init__()
        self.device = config.get(
            'device',
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.fast_adaptation_rate = float(config.get('fast_adaptation_rate', 0.05))
        self.max_adaptation_steps = int(config.get('max_adaptation_steps', 5))
        self.proximal_lambda = float(config.get('proximal_lambda', 0.05))
        self.l2_regularization = float(config.get('l2_regularization', 0.0001))
        self.epsilon = float(config.get('epsilon', 1e-8))
        self.gradient_clip_value = float(config.get('gradient_clip_value', 10.0))
        self.max_prototype_shift = float(config.get('max_prototype_shift', 5.0))
        self.enable_hard_mining = config.get('enable_hard_mining', True)
        self.hard_sample_quantile = float(config.get('hard_sample_quantile', 0.75))
        self.hard_sample_weight = float(config.get('hard_sample_weight', 1.5))
        self.use_adaptive_margin = config.get('use_adaptive_margin', True)
        self.base_margin = float(config.get('base_margin', 1.0))
        self.margin_scale = float(config.get('margin_scale', 1.5))
        self.enable_momentum = config.get('enable_momentum', True)
        self.momentum_beta = float(config.get('momentum_beta', 0.9))
        self.track_health = config.get('track_health', True)
        self.health_history = []
    def fast_evolve_for_episode(
            self,
            prototypes: Dict,
            support_features: torch.Tensor,
            support_labels: torch.Tensor,
            complexity_info: Dict,
            class_stats: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        evolution_config = self._extract_enhanced_evolution_config(complexity_info)
        class_centers = {}
        for cid, proto in prototypes.items():
            try:
                main_feature = self._extract_main_prototype(proto)
                class_centers[cid] = main_feature
            except Exception as e:
                pass
        evolved_prototypes = {}
        evolution_stats = {
            'per_class_stats': {},
            'global_health': 0.0,
            'convergence_info': {}
        }
        for class_id, proto_dict in prototypes.items():
            class_mask = (support_labels == class_id)
            class_features = support_features[class_mask]
            if class_features.numel() == 0:
                evolved_prototypes[class_id] = self._deep_copy_prototype(proto_dict)
                continue
            other_centers = self._extract_other_class_centers(class_centers, class_id)
            class_complexity = self._compute_class_complexity(
                class_features, complexity_info, class_id
            )
            adapted_config = self._adapt_config_for_class(
                evolution_config, class_complexity, len(class_features)
            )
            evolved_proto, class_stats_result = self._evolve_enhanced_class_prototypes(
                class_prototypes=proto_dict,
                class_features=class_features,
                class_labels=support_labels[class_mask],
                class_complexity_info=complexity_info,
                class_stats=class_stats,
                class_id=class_id,
                evolution_config=adapted_config,
                other_class_centers=other_centers
            )
            evolved_prototypes[class_id] = evolved_proto
            evolution_stats['per_class_stats'][class_id] = class_stats_result
        if self.track_health:
            evolution_stats['global_health'] = self._compute_global_health_score(
                evolution_stats['per_class_stats']
            )
            self.health_history.append(evolution_stats['global_health'])
        return evolved_prototypes, evolution_stats
    def _evolve_enhanced_class_prototypes(
            self,
            class_prototypes: Dict,
            class_features: torch.Tensor,
            class_labels: torch.Tensor,
            class_complexity_info: Dict,
            class_stats: Optional[Dict],
            class_id: int,
            evolution_config: Dict,
            other_class_centers: Optional[torch.Tensor] = None
    ) -> Tuple[Dict, Dict]:
        initial_main_prototype = self._extract_main_prototype(class_prototypes)
        evolved_main, main_stats = self._evolve_enhanced_main_prototype(
            initial_prototype=initial_main_prototype,
            class_features=class_features,
            evolution_config=evolution_config,
            adaptation_steps=evolution_config.get(
                'adaptation_steps', self.max_adaptation_steps
            ),
            other_class_centers=other_class_centers
        )
        evolved_proto = {
            'main': {
                'feature': evolved_main,
                'weight': class_prototypes.get('main', {}).get('weight', 1.0),
                'n_samples': len(class_features)
            },
            'sub': [],
            'complexity': class_prototypes.get('complexity', 0.5),
            'n_subclusters': 0
        }
        if 'class_complexity_info' in class_prototypes:
            evolved_proto['class_complexity_info'] = class_prototypes['class_complexity_info'].copy()
        if isinstance(class_prototypes, dict) and 'sub' in class_prototypes:
            initial_sub_prototypes = class_prototypes['sub']
            if initial_sub_prototypes and len(initial_sub_prototypes) > 0:
                evolved_subs, sub_stats = self._evolve_enhanced_sub_prototypes(
                    initial_sub_prototypes=initial_sub_prototypes,
                    target_features=class_features,
                    evolution_config=evolution_config,
                    main_prototype=evolved_main
                )
                evolved_proto['sub'] = evolved_subs
                evolved_proto['n_subclusters'] = len(evolved_subs)
            else:
                sub_stats = {'status': 'no_sub_prototypes'}
        else:
            sub_stats = {'status': 'no_sub_prototypes'}
        class_stats_result = {
            'main_stats': main_stats,
            'sub_stats': sub_stats,
            'n_iterations': main_stats.get('actual_steps', 0),
            'total_shift': main_stats.get('total_shift', 0.0)
        }
        return evolved_proto, class_stats_result
    def _evolve_enhanced_main_prototype(
            self,
            initial_prototype: torch.Tensor,
            class_features: torch.Tensor,
            evolution_config: Dict,
            adaptation_steps: int,
            other_class_centers: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        if not isinstance(initial_prototype, torch.Tensor):
            raise TypeError(f"Expected Tensor, got {type(initial_prototype)}")
        current_prototype = initial_prototype.clone().requires_grad_(True)
        N, D = class_features.shape
        evolution_rate = evolution_config.get('evolution_rate', self.fast_adaptation_rate)
        enable_hard_mining = evolution_config.get('enable_hard_mining', False)
        if evolution_config.get('use_momentum', False):
            momentum = torch.zeros_like(current_prototype)
            momentum_beta = self.momentum_beta
        else:
            momentum = None
        loss_history = []
        for step in range(adaptation_steps):
            target_mean = class_features.mean(dim=0)
            if class_features.size(0) > 1:
                target_var = class_features.var(dim=0, unbiased=False) + self.epsilon
            else:
                target_var = torch.ones_like(class_features[0]) * 0.1 + self.epsilon
            var_inv = 1.0 / target_var
            loss, gradient = self._compute_enhanced_evolution_loss_and_gradient(
                prototype=current_prototype,
                target_features=class_features,
                target_mean=target_mean,
                var_inv=var_inv,
                evolution_config=evolution_config,
                initial_prototype=initial_prototype,
                other_class_centers=other_class_centers,
                enable_hard_mining=enable_hard_mining,
                iteration=step
            )
            loss_history.append(loss.item())
            gradient = self._clip_gradient(gradient)
            if momentum is not None:
                momentum = momentum_beta * momentum + (1 - momentum_beta) * gradient
                effective_gradient = momentum
            else:
                effective_gradient = gradient
            current_prototype = current_prototype - evolution_rate * effective_gradient
            current_prototype = self._ensure_numerical_stability(current_prototype)
            if step >= 2 and self._check_convergence(loss_history[-3:]):
                break
        final_distance = torch.norm(
            class_features - current_prototype.unsqueeze(0), dim=1
        ).mean().item()
        total_shift = torch.norm(current_prototype - initial_prototype).item()
        convergence_quality = self._evaluate_convergence_quality(loss_history)
        final_health = self._compute_prototype_health_score(
            current_prototype, effective_gradient, loss_history[-1]
        )
        stats = {
            'final_loss': loss_history[-1] if loss_history else 0.0,
            'final_distance': final_distance,
            'total_shift': total_shift,
            'actual_steps': step + 1,
            'convergence_quality': convergence_quality,
            'final_health': final_health,
            'loss_history': loss_history
        }
        if not current_prototype.requires_grad:
            current_prototype = current_prototype.requires_grad_(True)
        return current_prototype, stats
    def _extract_main_prototype(self, prototype_data) -> torch.Tensor:
        if isinstance(prototype_data, dict):
            if 'main' in prototype_data:
                main_proto = prototype_data['main']
                if isinstance(main_proto, dict):
                    if 'feature' in main_proto:
                        feature = main_proto['feature']
                        if not isinstance(feature, torch.Tensor):
                            raise TypeError(f"'feature' should be Tensor, got {type(feature)}")
                        return feature
                    else:
                        raise KeyError(f"'main' dict missing 'feature': {main_proto.keys()}")
                elif isinstance(main_proto, torch.Tensor):
                    return main_proto
                else:
                    raise TypeError(f"'main' should be dict or Tensor, got {type(main_proto)}")
            elif 'feature' in prototype_data:
                feature = prototype_data['feature']
                if not isinstance(feature, torch.Tensor):
                    raise TypeError(f"'feature' should be Tensor, got {type(feature)}")
                return feature
            else:
                raise KeyError(f"Cannot extract feature from prototype, available keys: {list(prototype_data.keys())}")
        elif isinstance(prototype_data, torch.Tensor):
            return prototype_data
        else:
            raise TypeError(f"Unsupported prototype type: {type(prototype_data)}")
    def _compute_enhanced_evolution_loss_and_gradient(
            self,
            prototype: torch.Tensor,
            target_features: torch.Tensor,
            target_mean: torch.Tensor,
            var_inv: torch.Tensor,
            evolution_config: Dict,
            initial_prototype: torch.Tensor,
            other_class_centers: Optional[torch.Tensor] = None,
            enable_hard_mining: bool = False,
            iteration: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, D = target_features.shape
        diffs = prototype.unsqueeze(0) - target_features
        sq_dists = (diffs ** 2).sum(dim=1)
        if enable_hard_mining and N > 5:
            threshold = torch.quantile(sq_dists, self.hard_sample_quantile)
            hard_mask = sq_dists > threshold
            easy_mask = ~hard_mask
            weighted_sq = (diffs ** 2) * var_inv.unsqueeze(0)
            if hard_mask.any() and easy_mask.any():
                easy_loss = weighted_sq[easy_mask].mean()
                hard_loss = weighted_sq[hard_mask].mean()
                data_loss = 0.5 * easy_loss + self.hard_sample_weight * hard_loss
                grad_data_easy = 2.0 * (diffs[easy_mask] * var_inv.unsqueeze(0)).mean(dim=0)
                grad_data_hard = 2.0 * (diffs[hard_mask] * var_inv.unsqueeze(0)).mean(dim=0)
                grad_data = 0.5 * grad_data_easy + self.hard_sample_weight * grad_data_hard
            else:
                data_loss = weighted_sq.mean()
                grad_data = 2.0 * (diffs * var_inv.unsqueeze(0)).mean(dim=0)
        else:
            weighted_sq = (diffs ** 2) * var_inv.unsqueeze(0)
            data_loss = weighted_sq.mean()
            grad_data = 2.0 * (diffs * var_inv.unsqueeze(0)).mean(dim=0)
        prox_diff = prototype - initial_prototype
        prox_loss = self.proximal_lambda * torch.sum(prox_diff ** 2)
        grad_prox = 2.0 * self.proximal_lambda * prox_diff
        margin_loss = torch.tensor(0.0, device=prototype.device, dtype=prototype.dtype)
        grad_margin = torch.zeros_like(prototype)
        if other_class_centers is not None and other_class_centers.numel() > 0:
            if self.use_adaptive_margin:
                class_density = N / (target_features.size(0) + self.epsilon)
                margin = self.base_margin + self.margin_scale * (1.0 - class_density)
            else:
                margin = self.base_margin
            vecs = prototype.unsqueeze(0) - other_class_centers
            dists = torch.norm(vecs, dim=1, keepdim=True) + self.epsilon
            margin_violation = torch.relu(margin - dists.squeeze(1))
            margin_loss = 0.01 * margin_violation.mean()
            active = margin_violation > 0
            if active.any():
                active_vecs = vecs[active]
                active_dists = dists[active]
                grad_margin = -0.01 * (active_vecs / active_dists).mean(dim=0)
        total_loss = data_loss + prox_loss + margin_loss
        gradient = grad_data + grad_prox + grad_margin
        return total_loss, gradient
    def _evolve_enhanced_sub_prototypes(
            self,
            initial_sub_prototypes: List[Dict],
            target_features: torch.Tensor,
            evolution_config: Dict,
            main_prototype: torch.Tensor
    ) -> Tuple[List[Dict], Dict]:
        if not initial_sub_prototypes or len(initial_sub_prototypes) == 0:
            return [], {'status': 'no_sub_prototypes'}
        K = len(initial_sub_prototypes)
        N, D = target_features.shape
        if N < K:
            return initial_sub_prototypes, {'status': 'insufficient_samples'}
        try:
            centers = torch.stack([
                self._extract_feature(sub) for sub in initial_sub_prototypes
            ], dim=0)
        except Exception as e:
            return initial_sub_prototypes, {'status': 'extraction_failed'}
        dists = torch.cdist(target_features, centers, p=2)
        assign = torch.argmin(dists, dim=1)
        new_centers = []
        cluster_sizes = []
        for k in range(K):
            mask = (assign == k)
            if mask.any():
                new_center = target_features[mask].mean(dim=0)
                cluster_sizes.append(mask.sum().item())
            else:
                new_center = centers[k]
                cluster_sizes.append(0)
            new_centers.append(new_center)
        new_centers = torch.stack(new_centers, dim=0)
        move_rate = evolution_config.get('sub_prototype_move_rate', 0.3)
        centers = centers + move_rate * (new_centers - centers)
        centers = self._ensure_numerical_stability(centers)
        if not centers.requires_grad:
            centers = centers.requires_grad_(True)
        evolved_sub_prototypes = []
        for i in range(K):
            feature = centers[i]
            if not feature.requires_grad:
                feature = feature.requires_grad_(True)
            new_proto = {
                'feature': feature,
                'confidence': initial_sub_prototypes[i].get('confidence', 0.8),
                'complexity_weight': initial_sub_prototypes[i].get('complexity_weight', 1.0),
                'modal_type': initial_sub_prototypes[i].get('modal_type', 'mixed'),
                'source_complexity': initial_sub_prototypes[i].get('source_complexity', []),
                'sample_index': initial_sub_prototypes[i].get('sample_index', i),
                'weight': initial_sub_prototypes[i].get('weight', 1.0 / K),
                'n_samples': initial_sub_prototypes[i].get('n_samples', 1),
                'variance': initial_sub_prototypes[i].get('variance', 0.0),
                'cluster_size': cluster_sizes[i]
            }
            evolved_sub_prototypes.append(new_proto)
        original_centers = torch.stack([
            self._extract_feature(sub) for sub in initial_sub_prototypes
        ], dim=0)
        center_shift = torch.norm(centers - original_centers).item()
        stats = {
            'status': 'success',
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': sum(cluster_sizes) / K if K > 0 else 0.0,
            'empty_clusters': sum([1 for s in cluster_sizes if s == 0]),
            'center_shift': center_shift
        }
        return evolved_sub_prototypes, stats
    def _extract_enhanced_evolution_config(self, complexity_info: Dict) -> Dict:
        try:
            strategy = complexity_info.get('adaptation_strategy', {})
            global_complexity = complexity_info.get('global_complexity', 0.5)
            if torch.is_tensor(global_complexity):
                if global_complexity.dim() == 0:
                    global_complexity = global_complexity.item()
                else:
                    global_complexity = global_complexity.mean().item()
            global_complexity = float(global_complexity)
            complexity_factor = torch.sigmoid(
                torch.tensor(global_complexity - 0.5) * 2.0
            ).item()
            modal_complexities = complexity_info.get('modal_complexity', None)
            if modal_complexities is not None and torch.is_tensor(modal_complexities):
                if modal_complexities.dim() >= 2:
                    modal_std = modal_complexities.std(dim=1, unbiased=False).mean().item()
                else:
                    modal_std = 0.1
            else:
                modal_std = 0.1
            base_rate = strategy.get('evolution_rate', self.fast_adaptation_rate)
            adaptive_factor = 1.0 + complexity_factor * 0.5 + modal_std * 0.3
            config = {
                'evolution_rate': base_rate * adaptive_factor,
                'adaptive_rate_factor': complexity_factor,
                'modal_variance': modal_std,
                'use_momentum': strategy.get('level') in ['moderate', 'aggressive'],
                'use_adaptive_lr': strategy.get('level') == 'aggressive',
                'regularization_strength': self.l2_regularization * (1 + complexity_factor),
                'convergence_patience': 3 if strategy.get('level') == 'conservative' else 2,
                'max_evolution_magnitude': 2.0 if strategy.get('level') == 'aggressive' else 1.0,
                'modal_adaptive_factor': adaptive_factor,
                'enable_gradient_scaling': global_complexity > 0.8,
                'enable_momentum_decay': strategy.get('level') == 'aggressive',
                'adaptation_steps': strategy.get('adaptation_steps', self.max_adaptation_steps),
                'quality_threshold': 0.8 if strategy.get('level') == 'aggressive' else 0.6,
                'enable_hard_mining': global_complexity > 0.6,
                'adaptive_margin': self.base_margin + self.margin_scale * complexity_factor,
                'sub_prototype_move_rate': 0.2 + 0.2 * complexity_factor
            }
            return config
        except Exception as e:
            return self._get_default_evolution_config()
    def _get_default_evolution_config(self) -> Dict:
        return {
            'evolution_rate': self.fast_adaptation_rate,
            'adaptive_rate_factor': 0.5,
            'use_momentum': True,
            'use_adaptive_lr': False,
            'regularization_strength': self.l2_regularization,
            'convergence_patience': 3,
            'max_evolution_magnitude': 1.0,
            'modal_variance': 0.1,
            'modal_adaptive_factor': 1.1,
            'enable_gradient_scaling': False,
            'enable_momentum_decay': False,
            'adaptation_steps': 3,
            'quality_threshold': 0.6,
            'enable_hard_mining': False,
            'adaptive_margin': self.base_margin,
            'sub_prototype_move_rate': 0.3
        }
    def _adapt_config_for_class(
            self,
            base_config: Dict,
            class_complexity: float,
            num_samples: int
    ) -> Dict:
        config = base_config.copy()
        sample_factor = min(1.0, num_samples / 10.0)
        config['evolution_rate'] *= (0.5 + 0.5 * sample_factor)
        if class_complexity > 0.8:
            config['adaptation_steps'] = min(
                config['adaptation_steps'] + 2,
                self.max_adaptation_steps
            )
        elif class_complexity < 0.3:
            config['adaptation_steps'] = max(
                config['adaptation_steps'] - 1,
                1
            )
        if num_samples < 5:
            config['regularization_strength'] *= 1.5
            config['evolution_rate'] *= 0.7
        return config
    def _compute_class_complexity(
            self,
            class_features: torch.Tensor,
            complexity_info: Dict,
            class_id: int
    ) -> float:
        if isinstance(complexity_info, dict):
            if 'class_complexities' in complexity_info:
                class_complexities = complexity_info['class_complexities']
                if isinstance(class_complexities, dict) and class_id in class_complexities:
                    value = class_complexities[class_id]
                    if torch.is_tensor(value):
                        return value.item() if value.numel() == 1 else value.mean().item()
                    return float(value)
            if 'global_complexity' in complexity_info:
                global_comp = complexity_info['global_complexity']
                if torch.is_tensor(global_comp):
                    if global_comp.dim() == 0:
                        return global_comp.item()
                    else:
                        return global_comp.mean().item()
                return float(global_comp)
            if 'complexity_dict' in complexity_info:
                comp_dict = complexity_info['complexity_dict']
                if isinstance(comp_dict, dict):
                    if class_id in comp_dict:
                        value = comp_dict[class_id]
                        if torch.is_tensor(value):
                            return value.item() if value.numel() == 1 else value.mean().item()
                        return float(value)
        if class_features.numel() > 0 and class_features.size(0) > 1:
            variance = class_features.var(dim=0, unbiased=False).mean().item()
            normalized_complexity = min(1.0, variance / 10.0)
            return normalized_complexity
        return 0.5
    def _extract_other_class_centers(
            self,
            class_centers: Dict[int, torch.Tensor],
            exclude_class_id: int
    ) -> Optional[torch.Tensor]:
        other_centers = []
        for cid, center in class_centers.items():
            if cid != exclude_class_id:
                if torch.is_tensor(center):
                    other_centers.append(center)
        if not other_centers:
            return None
        try:
            return torch.stack(other_centers, dim=0)
        except Exception as e:
            return None
    def _extract_feature(self, prototype_dict: Dict) -> torch.Tensor:
        if 'feature' in prototype_dict:
            feature = prototype_dict['feature']
            if torch.is_tensor(feature):
                return feature
            else:
                raise TypeError(f"'feature' is not Tensor: {type(feature)}")
        else:
            raise KeyError(f"dict missing 'feature': {list(prototype_dict.keys())}")
    def _ensure_numerical_stability(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        original_requires_grad = tensor.requires_grad
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        tensor = torch.clamp(tensor, min=-self.gradient_clip_value, max=self.gradient_clip_value)
        if original_requires_grad and not tensor.requires_grad:
            tensor = tensor.requires_grad_(True)
        return tensor
    def _clip_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        grad_norm = torch.norm(gradient)
        if grad_norm > self.gradient_clip_value:
            gradient = gradient * (self.gradient_clip_value / (grad_norm + self.epsilon))
        return gradient
    def _check_convergence(self, recent_losses: List[float]) -> bool:
        if len(recent_losses) < 3:
            return False
        changes = [
            abs(recent_losses[i] - recent_losses[i - 1]) / (recent_losses[i - 1] + self.epsilon)
            for i in range(1, len(recent_losses))
        ]
        return all(change < 0.01 for change in changes)
    def _evaluate_convergence_quality(self, loss_history: List[float]) -> float:
        if len(loss_history) < 2:
            return 0.5
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        if initial_loss < self.epsilon:
            return 1.0
        improvement = (initial_loss - final_loss) / (initial_loss + self.epsilon)
        quality = min(1.0, max(0.0, improvement))
        return quality
    def _compute_prototype_health_score(
            self,
            prototype: torch.Tensor,
            gradient: torch.Tensor,
            loss: float
    ) -> float:
        grad_norm = torch.norm(gradient).item()
        grad_health = 1.0 / (1.0 + grad_norm)
        proto_norm = torch.norm(prototype).item()
        norm_health = 1.0 - abs(proto_norm - 1.0)
        norm_health = max(0.0, norm_health)
        loss_health = 1.0 / (1.0 + loss)
        health = 0.4 * grad_health + 0.3 * norm_health + 0.3 * loss_health
        return health
    def _compute_global_health_score(self, per_class_stats: Dict) -> float:
        if not per_class_stats:
            return 0.0
        health_scores = []
        for class_id, stats in per_class_stats.items():
            if 'main_stats' in stats and 'final_health' in stats['main_stats']:
                health_scores.append(stats['main_stats']['final_health'])
        if not health_scores:
            return 0.0
        return sum(health_scores) / len(health_scores)
    def _deep_copy_prototype(self, proto_dict: Dict) -> Dict:
        if not isinstance(proto_dict, dict):
            raise TypeError(f"Expected dict, got {type(proto_dict)}")
        copied = {}
        if 'main
