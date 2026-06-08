import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
logger = logging.getLogger(__name__)
class DynamicPrototypeGenerator(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(DynamicPrototypeGenerator, self).__init__()
        self.config = config
        self.device = config.get('device', torch.device('cpu'))
        self.feature_dim = config.get('feature_dim', 59)
        self.prototype_dim = config.get('prototype_dim', 128)
        self.adaptation_steps = config.get('adaptation_steps', 5)
        self.fast_adaptation_rate = config.get('fast_adaptation_rate', 0.05)
        self.use_query_adaptation = config.get('use_query_adaptation', True)
        self.query_adaptation_threshold = config.get('query_adaptation_threshold', 0.7)
        self.query_adaptation_steps = config.get('query_adaptation_steps', 3)
        self.epsilon = config.get('epsilon', 1e-8)
        self.clamp_range = config.get('clamp_range', (-1e6, 1e6))
        self.gradient_clip_value = config.get('gradient_clip_value', 10.0)
        self.use_batch_norm = config.get('use_batch_norm', True)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.feature_encoder = self._build_feature_encoder()
        self.use_prototype_projection = config.get('use_prototype_projection', False)
        self.prototype_proj_dim = config.get('prototype_proj_dim', 32)
        if self.use_prototype_projection:
            self.proto_proj = nn.Sequential(
                nn.Linear(self.prototype_dim, self.prototype_proj_dim),
                nn.LayerNorm(self.prototype_proj_dim)
            )
        else:
            self.proto_proj = None
        self._validate_config()
        self.use_complexity_estimator = config.get('use_complexity_estimator', True)
        if self.use_complexity_estimator:
            from models.complexity_estimator import ComplexityEstimator
            self.complexity_estimator = ComplexityEstimator(config)
            logger.info(" ")
        else:
            self.complexity_estimator = None
            logger.info(" ")
        self.use_hierarchy_constructor = config.get('use_hierarchy_constructor', True)
        if self.use_hierarchy_constructor:
            from models.hierarchy_constructor import HierarchyConstructor
            self.hierarchy_constructor = HierarchyConstructor(config)
            logger.info(" ")
        else:
            self.hierarchy_constructor = None
            logger.info(" ")
        self.use_evolution_engine = config.get('use_evolution_engine', True)
        if self.use_evolution_engine:
            from models.evolution_engine import PrototypeEvolutionEngine
            self.evolution_engine = PrototypeEvolutionEngine(config)
            logger.info(" ")
        else:
            self.evolution_engine = None
            logger.info(" ")
        self.enable_caching = config.get('enable_caching', True)
        self._complexity_cache: Dict[str, Any] = {}
        self._distance_cache: Optional[torch.Tensor] = None
        self._last_distance_cache_key: Optional[Tuple[int, int]] = None
        self._episode_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
    def _build_feature_encoder(self) -> nn.Module:
        use_deep = self.config.get('use_deep_encoder', True)
        if not use_deep:
            layers = []
            layers.append(nn.Linear(self.feature_dim, self.prototype_dim))
            if self.use_batch_norm:
                layers.append(nn.LayerNorm(self.prototype_dim))
            layers.append(nn.LeakyReLU(0.2))
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.Linear(self.prototype_dim, self.prototype_dim))
            layers.append(nn.LayerNorm(self.prototype_dim))
            layers.append(nn.LeakyReLU(0.2))
            encoder = nn.Sequential(*layers)
            self._init_encoder_weights(encoder)
            return encoder
        class ResidualBlock(nn.Module):
            def __init__(self, dim, dropout=0.03):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.LayerNorm(dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                    nn.LayerNorm(dim),
                )
                self.dropout = nn.Dropout(dropout)
                self.activation = nn.GELU()
            def forward(self, x):
                return self.activation(x + self.dropout(self.net(x)))
        class SEBlock(nn.Module):
            def __init__(self, dim, reduction=8):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(dim, dim // reduction),
                    nn.GELU(),
                    nn.Linear(dim // reduction, dim),
                    nn.Sigmoid()
                )
            def forward(self, x):
                attention = self.fc(x)
                return x * attention
        layers = []
        input_dim = self.feature_dim
        hidden_dim = self.config.get('encoder_hidden_dim', 512)
        output_dim = self.prototype_dim
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        ])
        num_blocks = self.config.get('encoder_num_blocks', 3)
        use_se = self.config.get('use_se_attention', True)
        for i in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim, self.dropout_rate))
            if use_se and i % 2 == 1:
                layers.append(SEBlock(hidden_dim))
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        ])
        layers.append(ResidualBlock(output_dim, self.dropout_rate))
        if use_se:
            layers.append(SEBlock(output_dim))
        layers.append(nn.LayerNorm(output_dim))
        encoder = nn.Sequential(*layers)
        self._init_deep_encoder_weights(encoder)
        return encoder
    def _init_deep_encoder_weights(self, encoder: nn.Module):
        for module in encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    def _init_encoder_weights(self, encoder: nn.Module):
        for module in encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    def _validate_config(self):
        required_fields = {
            'modal_indices': {'numerical': list(range(self.feature_dim))},
            'modal_dims': {'numerical': self.feature_dim},
            'modal_types': ['numerical']
        }
        for field, default_value in required_fields.items():
            if field not in self.config or not self.config[field]:
                logger.warning(f" '{field}'")
                self.config[field] = default_value
        self.config.setdefault('n_modalities', len(self.config['modal_types']))
        self.config.setdefault('modality_weights', [1.0 / self.config['n_modalities']] * self.config['n_modalities'])
        self.config.setdefault('complexity_epsilon', 1e-8)
        self.config.setdefault('weight_momentum', 0.9)
        self.config.setdefault('use_entropy_enhancement', True)
        self.config.setdefault('dataset_name', 'CICIDS2017')
        self.config.setdefault('verbose_mapping', False)
        self.config.setdefault('max_subclusters', 3)
        self.config.setdefault('min_samples_per_subcluster', 2)
        self.config.setdefault('subcluster_threshold', 0.5)
        self.config.setdefault('evolution_strategy', 'gradient_based')
        self.config.setdefault('use_meta_learning', True)
        self.config.setdefault('temperature', 1.0)
    def forward(self,
                support_data: torch.Tensor,
                support_labels: torch.Tensor,
                context_data: Optional[torch.Tensor] = None,
                context_labels: Optional[torch.Tensor] = None) -> Dict:
        self._episode_count += 1
        device = support_data.device
        unique_labels = torch.unique(support_labels)
        n_way = unique_labels.numel()
        support_features = self.feature_encoder(support_data)
        initial_prototypes: Dict[int, torch.Tensor] = {}
        for class_id in unique_labels.cpu().tolist():
            class_mask = (support_labels == class_id)
            class_feats = support_features[class_mask]
            if class_feats.numel() > 0:
                proto = class_feats.mean(dim=0)
            else:
                proto = torch.randn(self.prototype_dim, device=device) * 0.1
            if torch.allclose(proto, torch.zeros_like(proto), atol=1e-6):
                proto = torch.randn_like(proto) * 0.1 + 0.05
            initial_prototypes[class_id] = proto
        hier_proto = {
            cid: {'main': {'feature': proto, 'weight': 1.0}, 'sub': []}
            for cid, proto in initial_prototypes.items()
        }
        query_like = context_data if context_data is not None else support_data
        complexity_info = self._compute_or_retrieve_complexity(
            support_data=support_data,
            support_labels=support_labels,
            query_data=query_like
        )
        if self.use_hierarchy_constructor and self.hierarchy_constructor is not None:
            try:
                hier_result = self.hierarchy_constructor.build_hierarchy(
                    support_features=support_features,
                    complexity_info=complexity_info,
                    labels=support_labels,
                    class_stats=None,
                    accuracy_feedback_per_class=None
                )
                hier_proto = hier_result
                logger.debug("")
            except Exception as e:
                logger.warning(f": {e}")
                hier_proto = {
                    cid: {'main': {'feature': proto, 'weight': 1.0}, 'sub': []}
                    for cid, proto in initial_prototypes.items()
                }
        else:
            logger.debug("")
        if self.use_evolution_engine and self.evolution_engine is not None:
            try:
                evolved_prototypes, evolution_stats = self.evolution_engine.fast_evolve_for_episode(
                    prototypes=hier_proto,
                    support_features=support_features,
                    support_labels=support_labels,
                    complexity_info=complexity_info,
                    class_stats=None
                )
                logger.debug("")
            except Exception as e:
                logger.warning(f": {e}")
                evolved_prototypes = hier_proto
                evolution_stats = {}
        else:
            evolved_prototypes = hier_proto
            evolution_stats = {}
            logger.debug("")
        evolved_prototypes = self._sanitize_evolved_prototypes(evolved_prototypes, initial_prototypes)
        if self.use_query_adaptation and context_data is not None and context_data.numel() > 0:
            query_features = self.feature_encoder(context_data)
            proto_info_for_adapt = self._build_unified_prototype_info(
                evolved_prototypes=evolved_prototypes,
                query_features=query_features,
                support_labels=support_labels
            )
            refined = self._perform_query_adaptation(
                evolved_prototypes=evolved_prototypes,
                query_features=query_features,
                initial_prototype_info=proto_info_for_adapt,
                complexity_info=complexity_info
            )
            evolved_prototypes = refined['refined_prototypes']
            target_queries = query_features
        else:
            target_queries = support_features
        prototype_info = self._build_unified_prototype_info(
            evolved_prototypes=evolved_prototypes,
            query_features=target_queries,
            support_labels=support_labels
        )
        if self.proto_proj is not None:
            proj_protos = self._apply_proto_projection(prototype_info['prototypes_tensor'])
            proj_queries = self._apply_proto_projection(target_queries)
            distances = self._compute_stable_distances(proj_queries, proj_protos)
            prototype_info['prototypes_tensor'] = proj_protos
            prototype_info['distances'] = distances
            prototype_info['class_logits'] = -distances
        formatted_prototypes: Dict[int, Dict[str, Any]] = {}
        for class_id, proto in evolved_prototypes.items():
            if isinstance(proto, dict):
                formatted_prototypes[class_id] = proto
            else:
                formatted_prototypes[class_id] = {'main': {'feature': proto, 'weight': 1.0}}
        return {
            'evolved_prototypes': formatted_prototypes,
            'attention_weights': {},
            'auxiliary_info': {
                'n_way': n_way,
                'feature_dim': self.prototype_dim,
                'label_order': prototype_info.get('label_order')
            },
            'prototype_info': prototype_info
        }
    def _sanitize_evolved_prototypes(self,
                                     evolved_prototypes: Dict[int, Dict[str, Any]],
                                     fallback_initial: Dict[int, torch.Tensor]) -> Dict[int, Dict[str, Any]]:
        sanitized = {}
        for cid, pdata in evolved_prototypes.items():
            try:
                main = self._extract_main_prototype(pdata)
                bad = (not torch.isfinite(main).all()) or (main.norm().item() < 1e-6)
                if bad:
                    repl = fallback_initial.get(cid, None)
                    if repl is None or repl.norm().item() < 1e-6:
                        repl = torch.randn(self.prototype_dim, device=main.device) * 0.1 + 0.05
                    if isinstance(pdata, dict):
                        new_pd = {k: (v.copy() if isinstance(v, dict) else v) for k, v in pdata.items()}
                        self._update_main_prototype(new_pd, repl)
                        sanitized[cid] = new_pd
                    else:
                        sanitized[cid] = {'main': {'feature': repl, 'weight': 1.0}}
                else:
                    sanitized[cid] = pdata
            except Exception:
                repl = fallback_initial.get(cid, torch.randn(self.prototype_dim, device=self.device) * 0.1 + 0.05)
                sanitized[cid] = {'main': {'feature': repl, 'weight': 1.0}}
        return sanitized
    def _extract_main_prototype(self, prototype_data: Any) -> torch.Tensor:
        if isinstance(prototype_data, dict):
            if 'main' in prototype_data:
                main_proto = prototype_data['main']
                if isinstance(main_proto, dict) and 'feature' in main_proto:
                    return main_proto['feature']
                elif isinstance(main_proto, torch.Tensor):
                    return main_proto
            elif 'feature' in prototype_data:
                return prototype_data['feature']
        elif isinstance(prototype_data, torch.Tensor):
            return prototype_data
        raise ValueError(f": {type(prototype_data)}")
    def _update_main_prototype(self, prototype_data: Any, new_feature: torch.Tensor):
        if isinstance(prototype_data, dict):
            if 'main' in prototype_data:
                if isinstance(prototype_data['main'], dict):
                    prototype_data['main']['feature'] = new_feature
                else:
                    prototype_data['main'] = new_feature
            elif 'feature' in prototype_data:
                prototype_data['feature'] = new_feature
    def _compute_or_retrieve_complexity(self, support_data: torch.Tensor,
                                        support_labels: torch.Tensor,
                                        query_data: torch.Tensor) -> Dict[str, Any]:
        if not self.use_complexity_estimator or self.complexity_estimator is None:
            return self._get_default_complexity_info(support_data, support_labels, query_data)
        if not self.enable_caching:
            return self.complexity_estimator(support_data, support_labels, query_data)
        cache_key = self._generate_complexity_cache_key(support_data, support_labels, query_data)
        if cache_key in self._complexity_cache:
            self._cache_hits += 1
            return self._complexity_cache[cache_key]
        self._cache_misses += 1
        complexity_info = self.complexity_estimator(support_data, support_labels, query_data)
        self._complexity_cache[cache_key] = complexity_info
        if len(self._complexity_cache) > 100:
            for _ in range(10):
                self._complexity_cache.pop(next(iter(self._complexity_cache)))
        return complexity_info
    def _get_default_complexity_info(self, support_data: torch.Tensor,
                                    support_labels: torch.Tensor,
                                    query_data: torch.Tensor) -> Dict[str, Any]:
        device = support_data.device
        n_support = support_data.size(0)
        n_query = query_data.size(0)
        unique_labels = torch.unique(support_labels)
        n_way = unique_labels.numel()
        default_complexity = 0.5
        global_complexity = torch.full((n_support + n_query,), default_complexity, device=device)
        modal_complexity = torch.full((n_support + n_query, 3), default_complexity, device=device)
        adaptation_strategy = {
            'level': 'moderate',
            'evolution_rate': 0.05,
            'adaptation_steps': 3
        }
        return {
            'modal_complexity': modal_complexity,
            'global_complexity': global_complexity,
            'adaptation_strategy': adaptation_strategy,
            'n_way': n_way,
            'n_support': n_support,
            'n_query': n_query
        }
    def _generate_complexity_cache_key(self, support_data: torch.Tensor,
                                       support_labels: torch.Tensor,
                                       query_data: torch.Tensor) -> str:
        s_mean, s_std = support_data.mean().item(), support_data.std().item()
        q_mean, q_std = query_data.mean().item(), query_data.std().item()
        unique_labels, counts = torch.unique(support_labels, return_counts=True)
        label_signature = f"{unique_labels.tolist()}_{counts.tolist()}"
        return f"{s_mean:.4f}_{s_std:.4f}_{q_mean:.4f}_{q_std:.4f}_{label_signature}"
    def _get_cache_hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0
    def _perform_unified_evolution(self, hierarchical_prototypes, support_features,
                                   support_labels, complexity_info, adaptation_steps):
        steps = adaptation_steps if adaptation_steps is not None else self.adaptation_steps
        evolved_prototypes, evolution_stats = self.evolution_engine.fast_evolve_for_episode(
            prototypes=hierarchical_prototypes,
            support_features=support_features,
            support_labels=support_labels,
            complexity_info=complexity_info,
            class_stats=None
        )
        return {
            'evolved_prototypes': evolved_prototypes,
            'evolution_stats': evolution_stats
        }
    def _perform_query_adaptation(self, evolved_prototypes: Dict[int, Dict[str, Any]],
                                  query_features: torch.Tensor,
                                  initial_prototype_info: Dict[str, Any],
                                  complexity_info: Dict[str, Any]) -> Dict[str, Any]:
        refined_prototypes = {}
        logits = initial_prototype_info['class_logits']
        pred_labels = logits.argmax(dim=1)
        confidence = torch.softmax(logits, dim=1).max(dim=1)[0]
        lr = self.fast_adaptation_rate * 0.5
        for class_id, proto_data in evolved_prototypes.items():
            class_mask = (pred_labels == class_id) & (confidence > 0.8)
            if class_mask.any():
                class_queries = query_features[class_mask]
                current_main = self._extract_main_prototype(proto_data)
                query_center = class_queries.mean(dim=0)
                refined_main = current_main + lr * (query_center - current_main)
                refined_prototypes[class_id] = proto_data.copy() if isinstance(proto_data, dict) else {}
                self._update_main_prototype(refined_prototypes[class_id], refined_main)
            else:
                refined_prototypes[class_id] = proto_data
        return {'refined_prototypes': refined_prototypes}
    def _build_unified_prototype_info(self, evolved_prototypes: Dict[int, Dict[str, Any]],
                                      query_features: torch.Tensor,
                                      support_labels: torch.Tensor) -> Dict[str, Any]:
        organized = self._extract_and_organize_prototypes(evolved_prototypes, len(evolved_prototypes))
        main_prototypes = organized['main_prototypes']
        sub_prototypes = organized.get('sub_prototypes')
        label_order = organized.get('label_order')
        distances_main = self._compute_stable_distances(query_features, main_prototypes)
        class_logits = -distances_main
        if sub_prototypes is not None and sub_prototypes.numel() > 0:
            distances_sub = self._compute_stable_distances(query_features, sub_prototypes)
            min_sub_distances, _ = distances_sub.min(dim=1, keepdim=True)
            class_logits = class_logits - 0.1 * min_sub_distances
        return {
            'prototypes_tensor': main_prototypes,
            'class_logits': class_logits,
            'distances': distances_main,
            'n_way': len(evolved_prototypes),
            'label_order': label_order
        }
    def _extract_and_organize_prototypes(self, hierarchical_prototypes: Dict[int, Dict[str, Any]],
                                         n_way: int) -> Dict[str, Any]:
        main_prototypes_list: List[torch.Tensor] = []
        sub_prototypes_list: List[torch.Tensor] = []
        label_order = sorted(hierarchical_prototypes.keys())
        for class_id in label_order:
            if class_id not in hierarchical_prototypes:
                logger.warning(f" {class_id} hierarchical_prototypes")
                continue
            proto_data = hierarchical_prototypes[class_id]
            main_proto = self._extract_main_prototype(proto_data)
            main_prototypes_list.append(main_proto)
            if isinstance(proto_data, dict) and 'sub' in proto_data:
                sub_protos = proto_data['sub']
                if isinstance(sub_protos, list) and len(sub_protos) > 0:
                    for sub in sub_protos:
                        sub_feature = self._extract_feature(sub)
                        if sub_feature is not None:
                            sub_prototypes_list.append(sub_feature)
        main_prototypes = torch.stack(main_prototypes_list) if main_prototypes_list else None
        sub_prototypes = torch.stack(sub_prototypes_list) if sub_prototypes_list else None
        return {
            'main_prototypes': main_prototypes,
            'sub_prototypes': sub_prototypes,
            'label_order': label_order
        }
    def _extract_feature(self, prototype_data: Any) -> Optional[torch.Tensor]:
        if isinstance(prototype_data, dict):
            if 'feature' in prototype_data:
                return prototype_data['feature']
            elif 'main' in prototype_data:
                return self._extract_feature(prototype_data['main'])
        elif isinstance(prototype_data, torch.Tensor):
            return prototype_data
        return None
    def _compute_stable_distances(self, query_features: torch.Tensor,
                                  prototypes_tensor: torch.Tensor) -> torch.Tensor:
        cache_key = (id(query_features), id(prototypes_tensor))
        if hasattr(self, '_last_distance_cache_key') and self._last_distance_cache_key == cache_key:
            return self._distance_cache
        distances = torch.cdist(query_features, prototypes_tensor, p=2)
        distances = torch.clamp(distances, min=self.epsilon)
        self._distance_cache = distances
        self._last_distance_cache_key = cache_key
        return distances
    def _apply_proto_projection(self, prototypes_tensor: torch.Tensor) -> torch.Tensor:
        if self.proto_proj is None:
            return prototypes_tensor
        projected = self.proto_proj(prototypes_tensor)
        projected = F.normalize(projected, p=2, dim=1)
        return projected
    def _build_unified_result(self, query_features: torch.Tensor,
                              prototype_info: Dict[str, Any],
                              evolved_prototypes: Dict[int, Dict[str, Any]],
                              complexity_info: Dict[str, Any],
                              evolution_stats: Dict[str, Any],
                              return_episode_info: bool) -> Dict[str, Any]:
        result = {
            'query_features': query_features,
            'prototype_info': prototype_info,
            'evolved_prototypes': evolved_prototypes
        }
        if return_episode_info:
            result['episode_info'] = {
                'complexity_info': complexity_info,
                'evolution_stats': evolution_stats,
                'episode_id': self._episode_count,
                'cache_hit_rate': self._get_cache_hit_rate()
            }
        return result
    def _ensure_numerical_stability(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.requires_grad:
            stable_tensor = tensor.detach()
        else:
            stable_tensor = tensor
        mask_nan = torch.isnan(stable_tensor)
        mask_inf = torch.isinf(stable_tensor)
        has_invalid = mask_nan.any() or mask_inf.any()
        if has_invalid:
            stable_tensor = torch.where(mask_nan, torch.zeros_like(stable_tensor), stable_tensor)
            stable_tensor = torch.where(mask_inf, torch.sign(stable_tensor) * 1e6, stable_tensor)
        stable_tensor = torch.clamp(stable_tensor, min=self.clamp_range[0], max=self.clamp_range[1])
        return stable_tensor
    def estimate_task_difficulty(self, support_data: torch.Tensor,
                                 support_labels: torch.Tensor) -> Dict[str, Any]:
        complexity_info = self.complexity_estimator(support_data, support_labels, support_data)
        difficulty_score = complexity_info.get('global_complexity', torch.tensor(0.5))
        if isinstance(difficulty_score, torch.Tensor):
            if difficulty_score.numel() > 1:
                difficulty_score = difficulty_score.mean().item()
            else:
                difficulty_score = difficulty_score.item()
        if difficulty_score < 0.3:
            difficulty_level = 'easy'
            recommended_steps = 3
        elif difficulty_score < 0.7:
            difficulty_level = 'moderate'
            recommended_steps = 5
        else:
            difficulty_level = 'hard'
            recommended_steps = 8
        return {
            'difficulty_score': difficulty_score,
            'difficulty_level': difficulty_level,
            'recommended_steps': recommended_steps,
            'complexity_info': complexity_info
        }
    def get_prototype_summary(self, evolved_prototypes: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        summary = {
            'n_classes': len(evolved_prototypes),
            'class_details': {}
        }
        for class_id, proto_data in evolved_prototypes.items():
            class_info = {
                'has_main': False,
                'n_sub_prototypes': 0,
                'main_norm': 0.0
            }
            try:
                main_proto = self._extract_main_prototype(proto_data)
                class_info['has_main'] = True
                class_info['main_norm'] = torch.norm(main_proto).item()
            except:
                pass
            if isinstance(proto_data, dict) and 'sub' in proto_data:
                sub_protos = proto_data['sub']
                if isinstance(sub_protos, list):
                    class_info['n_sub_prototypes'] = len(sub_protos)
            summary['class_details'][class_id] = class_info
        return summary
    def reset_cache(self):
        self._complexity_cache.clear()
        self._distance_cache = None
        self._last_distance_cache_key = None
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("")
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'episode_count': self._episode_count,
            'cache_hit_rate': self._get_cache_hit_rate(),
            'cache_size': len(self._complexity_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses
        }
    def visualize_prototypes(self, evolved_prototypes: Dict[int, Dict[str, Any]],
                             method: str = 'tsne') -> Dict[str, Any]:
        all_features = []
        all_labels = []
        all_types = []
        for class_id, proto_data in evolved_prototypes.items():
            try:
                main_proto = self._extract_main_prototype(proto_data)
                all_features.append(main_proto.detach().cpu())
                all_labels.append(class_id)
                all_types.append('main')
            except:
                pass
            if isinstance(proto_data, dict) and 'sub' in proto_data:
                sub_protos = proto_data['sub']
                if isinstance(sub_protos, list):
                    for sub in sub_protos:
                        sub_feature = self._extract_feature(sub)
                        if sub_feature is not None:
                            all_features.append(sub_feature.detach().cpu())
                            all_labels.append(class_id)
                            all_types.append('sub')
        if not all_features:
            return {'coords': None, 'labels': [], 'types': []}
        features_matrix = torch.stack(all_features).numpy()
        if method == 'tsne':
            from sklearn.manifold import TSNE
            coords = TSNE(n_components=2, random_state=42).fit_transform(features_matrix)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            coords = PCA(n_components=2).fit_transform(features_matrix)
        else:
            raise ValueError(f": {method}")
        return {
            'coords': coords,
            'labels': all_labels,
            'types': all_types
        }
