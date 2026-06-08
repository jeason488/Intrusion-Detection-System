import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import os
import logging
from torch.utils.checkpoint import checkpoint
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
class DHPENNetwork(nn.Module):
    def __init__(self, config: dict):
        super(DHPENNetwork, self).__init__()
        if isinstance(config.get('device'), str):
            device_str = config['device']
            if device_str == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif device_str == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
                self.device = torch.device('xpu')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = config.get('device', torch.device('cpu'))
        self.config = config
        self.n_way = config.get('n_way', 5)
        self.k_shot = config.get('k_shot', 5)
        self.n_query = config.get('n_query', 15)
        self.inner_lr = config.get('inner_lr', 0.001)
        self.meta_lr = config.get('meta_lr', 0.0001)
        self.adaptation_steps = config.get('adaptation_steps', 5)
        self.epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.max_weight_norm = 10.0
        self.input_clip_value = 100.0
        self.use_gradient_checkpointing = config.get('use_gradient_checkpointing', False)
        self.alpha_fusion = config.get('alpha_fusion', 0.7)
        self.beta_update = config.get('beta_update', 0.9)
        self.use_global_memory = config.get('use_global_memory', True)
        init_temperature = config.get('temperature', 10.0)
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(init_temperature), dtype=torch.float32)
        )
        self.temperature_min = 0.5
        self.temperature_max = 20.0
        self._init_modules(config)
        self.apply(self._init_weights_recursively)
        self._verify_initialization()
        self.global_prototype_memory = defaultdict(lambda: None)
        self.memory_update_count = 0
        self._episode_count = 0
        self._memory_hit_count = 0
        self._memory_miss_count = 0
        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.memory_save_interval = config.get('memory_save_interval', 100)
        self._first_linear_layer = None
        self._cache_first_linear()
    def forward(self, support_data, support_labels, query_data):
        if self.training:
            inner_result = self.meta_inner_loop(
                support_data, support_labels,
                query_data, query_labels=None,
                return_info=False
            )
            fast_weights = inner_result['fast_weights']
            task_prototypes = inner_result['task_prototypes']
            query_features = self._functional_forward(query_data, fast_weights)
            logits = self._classify_with_prototypes(query_features, task_prototypes)
        else:
            with torch.no_grad():
                generation_result = self.prototype_generator(
                    support_data, support_labels,
                    support_data, support_labels
                )
                task_prototypes = generation_result['evolved_prototypes']
                task_prototypes = self._normalize_prototypes(task_prototypes)
                if self.use_global_memory:
                    task_prototypes = self._fuse_global_memory(task_prototypes)
                query_features = self.prototype_generator.feature_encoder(query_data)
                logits = self._classify_with_prototypes(query_features, task_prototypes)
        return logits
    def _init_modules(self, config: dict):
        from models.dynamic_prototype_generator import DynamicPrototypeGenerator
        self.prototype_generator = DynamicPrototypeGenerator(config)
        logger.info("  DynamicPrototypeGenerator")
        from models.losses import EnhancedMetaLoss
        self.loss_fn = EnhancedMetaLoss(config)
        logger.info("  EnhancedMetaLoss")
    @property
    def temperature(self) -> torch.Tensor:
        temp = torch.exp(self.log_temperature)
        return torch.clamp(temp, min=self.temperature_min, max=self.temperature_max)
    def _normalize_prototypes(self, prototypes: dict) -> dict:
        normalized = {}
        proto_dim = self.config.get('prototype_dim', 128)
        for class_id, proto_data in prototypes.items():
            try:
                if isinstance(proto_data, dict):
                    if 'main' in proto_data:
                        main_proto = proto_data['main']
                        if isinstance(main_proto, dict) and 'feature' in main_proto:
                            main_tensor = main_proto['feature']
                        else:
                            main_tensor = main_proto
                    else:
                        logger.warning(f" {class_id}'main': {proto_data.keys()}")
                        main_tensor = torch.randn(proto_dim, device=self.device) * 0.1
                elif torch.is_tensor(proto_data):
                    main_tensor = proto_data
                else:
                    logger.error(f" {class_id}: {type(proto_data)}")
                    main_tensor = torch.randn(proto_dim, device=self.device) * 0.1
                if not torch.is_tensor(main_tensor):
                    logger.error(f" {class_id}Tensor: {type(main_tensor)}")
                    main_tensor = torch.randn(proto_dim, device=self.device) * 0.1
                if main_tensor.dim() == 0:
                    main_tensor = main_tensor.unsqueeze(0).repeat(proto_dim)
                elif main_tensor.dim() == 2:
                    main_tensor = main_tensor.mean(dim=0)
                if main_tensor.size(0) != proto_dim:
                    if main_tensor.size(0) < proto_dim:
                        padding = torch.zeros(proto_dim - main_tensor.size(0),
                                              device=main_tensor.device)
                        main_tensor = torch.cat([main_tensor, padding])
                    else:
                        main_tensor = main_tensor[:proto_dim]
                if torch.abs(main_tensor).sum() < 1e-6:
                    logger.warning(f" {class_id}0")
                    main_tensor = torch.randn(proto_dim, device=self.device) * 0.3
                    main_tensor += 0.1
                norm = main_tensor.norm(p=2) + self.epsilon
                main_tensor = main_tensor / norm
                if torch.isnan(main_tensor).any() or torch.isinf(main_tensor).any():
                    logger.error(f" {class_id}NaN/Inf")
                    main_tensor = torch.randn(proto_dim, device=self.device) * 0.01
                    main_tensor = main_tensor / (main_tensor.norm(p=2) + self.epsilon)
                normalized[class_id] = main_tensor
            except Exception as e:
                logger.error(f" {class_id}: {e}")
                import traceback
                traceback.print_exc()
                normalized[class_id] = torch.randn(proto_dim, device=self.device) * 0.01
        return normalized
    def meta_inner_loop(
            self,
            support_data: torch.Tensor,
            support_labels: torch.Tensor,
            query_data: torch.Tensor = None,
            query_labels: torch.Tensor = None,
            return_info: bool = False
    ) -> dict:
        try:
            with torch.no_grad():
                generation_result = self.prototype_generator(
                    support_data, support_labels,
                    support_data, support_labels
                )
            if generation_result is None:
                raise ValueError(" prototype_generator  None")
            evolved_prototypes = generation_result.get('evolved_prototypes', {})
            normalized_prototypes = self._normalize_prototypes(evolved_prototypes)
            for cid, proto in normalized_prototypes.items():
                if torch.abs(proto).sum() < 1e-6:
                    logger.error(f" {cid}0!")
                if torch.isnan(proto).any():
                    logger.error(f" {cid}NaN!")
            fused_prototypes = self._fuse_global_memory(normalized_prototypes)
            encoder = self.prototype_generator.feature_encoder
            first_linear = None
            for module in encoder.modules():
                if isinstance(module, nn.Linear):
                    first_linear = module
                    break
            if first_linear is None:
                raise RuntimeError(" Linear")
            if self._first_linear_layer is None:
                raise RuntimeError(" ")
            first_linear = self._first_linear_layer
            fast_weight = first_linear.weight.clone().detach().requires_grad_(True)
            fast_bias = first_linear.bias.clone().detach().requires_grad_(True) \
                if first_linear.bias is not None else None
            fast_weights = [fast_weight]
            if fast_bias is not None:
                fast_weights.append(fast_bias)
            if torch.isnan(fast_weight).any():
                logger.error(" NaN!")
                fast_weight.data = torch.randn_like(fast_weight) * 0.01
            inner_losses = []
            for step in range(self.adaptation_steps):
                support_features = self._functional_forward(support_data, fast_weights)
                if torch.isnan(support_features).any():
                    logger.error(f" {step}: NaN,")
                    break
                logits = self._classify_with_prototypes(support_features, fused_prototypes)
                loss_result = self.loss_fn(
                    logits, support_labels,
                    {'evolved_prototypes': fused_prototypes}
                )
                inner_loss = loss_result['total_loss']
                inner_losses.append(inner_loss.item())
                if torch.isnan(inner_loss) or torch.isinf(inner_loss):
                    logger.error(f" {step}: ={inner_loss.item()}")
                    break
                is_last_step = (step == self.adaptation_steps - 1)
                grads = torch.autograd.grad(
                    outputs=inner_loss,
                    inputs=fast_weights,
                    create_graph=self.training,
                    retain_graph=not is_last_step,
                    allow_unused=False
                )
                new_fast_weights = []
                for param, grad in zip(fast_weights, grads):
                    if grad is None:
                        logger.warning(" None,")
                        new_fast_weights.append(param)
                        continue
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        logger.error(f" NaN/Inf,={grad.norm().item():.4f}")
                        new_fast_weights.append(param)
                        continue
                    grad_norm = grad.norm().item()
                    if grad_norm > self.max_grad_norm:
                        grad = grad * (self.max_grad_norm / (grad_norm + self.epsilon))
                    updated_param = param - self.inner_lr * grad
                    if torch.isnan(updated_param).any():
                        logger.error(" NaN!")
                        new_fast_weights.append(param)
                        continue
                    param_norm = updated_param.norm().item()
                    if param_norm > self.max_weight_norm:
                        updated_param = updated_param * (self.max_weight_norm / (param_norm + self.epsilon))
                    updated_param = updated_param.detach().requires_grad_(True)
                    new_fast_weights.append(updated_param)
                fast_weights = new_fast_weights
            result = {
                'fast_weights': fast_weights,
                'task_prototypes': fused_prototypes,
                'inner_losses': inner_losses,
                'final_loss': inner_losses[-1] if inner_losses else float('inf'),
            }
            if return_info and query_data is not None and query_labels is not None:
                with torch.no_grad():
                    query_features = self._functional_forward(query_data, fast_weights)
                    val_logits = self._classify_with_prototypes(query_features, fused_prototypes)
                    val_acc = self._compute_accuracy(val_logits, query_labels)
                    result['metrics'] = {'inner_validation_acc': val_acc}
                    result['adaptation_quality'] = self._assess_adaptation_quality(inner_losses)
            return result
        except Exception as e:
            logger.error(f" meta_inner_loop : {e}")
            import traceback
            traceback.print_exc()
            encoder = self.prototype_generator.feature_encoder
            first_linear = None
            for module in encoder.modules():
                if isinstance(module, nn.Linear):
                    first_linear = module
                    break
            if first_linear is None:
                raise RuntimeError(" Linear")
            safe_weights = [
                first_linear.weight.clone().detach().requires_grad_(True),
                first_linear.bias.clone().detach().requires_grad_(True) if first_linear.bias is not None else None
            ]
            safe_weights = [w for w in safe_weights if w is not None]
            return {
                'fast_weights': safe_weights,
                'task_prototypes': {},
                'inner_losses': [float('inf')],
                'final_loss': float('inf'),
            }
    def _functional_forward(self, x: torch.Tensor, fast_weights: list) -> torch.Tensor:
        x = torch.clamp(x, min=-self.input_clip_value, max=self.input_clip_value)
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        encoder = self.prototype_generator.feature_encoder
        children = list(encoder.children())
        first_idx = None
        for i, m in enumerate(children):
            if isinstance(m, nn.Linear):
                first_idx = i
                break
        if first_idx is None:
            raise RuntimeError("encoder  Linear")
        w = fast_weights[0]
        b = fast_weights[1] if len(fast_weights) > 1 else None
        if torch.isnan(w).any(): w = torch.nan_to_num(w, nan=0.0)
        if b is not None and torch.isnan(b).any(): b = torch.nan_to_num(b, nan=0.0)
        x = F.linear(x, w, b)
        for m in children[first_idx + 1:]:
            x = m(x)
            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)
        return x
    def meta_outer_loop(
            self,
            query_data: torch.Tensor,
            query_labels: torch.Tensor,
            fast_weights: list,
            task_prototypes: dict
    ) -> tuple:
        query_features = self._functional_forward(query_data, fast_weights)
        logits = self._classify_with_prototypes(query_features, task_prototypes)
        loss_result = self.loss_fn(
            logits, query_labels,
            {'evolved_prototypes': task_prototypes}
        )
        meta_loss = loss_result['total_loss']
        with torch.no_grad():
            accuracy = self._compute_accuracy(logits, query_labels)
            predictions = torch.argmax(logits, dim=1)
            unique_labels = torch.unique(query_labels)
            per_class_acc = {}
            for label in unique_labels:
                mask = (query_labels == label)
                if mask.sum() > 0:
                    class_acc = (predictions[mask] == query_labels[mask]).float().mean().item()
                    per_class_acc[label.item()] = class_acc
            probs = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidence_stats = {
                'mean': max_probs.mean().item(),
                'std': max_probs.std().item()
            }
        metrics = {
            'meta_loss': meta_loss.item(),
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'confidence': confidence_stats,
            'loss_components': loss_result
        }
        return meta_loss, metrics
    def meta_train_step(
            self,
            support_data: torch.Tensor,
            support_labels: torch.Tensor,
            query_data: torch.Tensor,
            query_labels: torch.Tensor,
            meta_optimizer: torch.optim.Optimizer
    ) -> dict:
        self.train()
        inner_result = self.meta_inner_loop(
            support_data, support_labels,
            query_data, query_labels,
            return_info=True
        )
        fast_weights = inner_result['fast_weights']
        task_prototypes = inner_result['task_prototypes']
        inner_losses = inner_result['inner_losses']
        meta_optimizer.zero_grad()
        meta_loss, outer_metrics = self.meta_outer_loop(
            query_data, query_labels,
            fast_weights,
            task_prototypes
        )
        meta_loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=self.gradient_clip
        )
        meta_optimizer.step()
        if self.use_global_memory:
            self.update_global_memory(task_prototypes)
        self._episode_count += 1
        if self._episode_count % self.memory_save_interval == 0:
            self.save_global_memory()
        metrics = {
            'episode': self._episode_count,
            'meta_loss': meta_loss.item(),
            'accuracy': outer_metrics['accuracy'],
            'inner_losses': inner_losses,
            'final_inner_loss': inner_losses[-1] if inner_losses else 0.0,
            'adaptation_steps': len(inner_losses),
            'gradient_norm': total_norm.item(),
            'temperature': self.temperature.item(),
            'per_class_accuracy': outer_metrics['per_class_accuracy'],
            'confidence': outer_metrics['confidence'],
            'global_memory_size': len(self.global_prototype_memory),
            'adaptation_quality': inner_result.get('adaptation_quality', 0.0)
        }
        return metrics
    def meta_test_step(
            self,
            support_data: torch.Tensor,
            support_labels: torch.Tensor,
            query_data: torch.Tensor,
            query_labels: torch.Tensor = None,
            use_global_memory: bool = True
    ) -> dict:
        self.eval()
        with torch.no_grad():
            generation_result = self.prototype_generator(
                support_data, support_labels,
                support_data, support_labels
            )
            task_prototypes = generation_result['evolved_prototypes']
            task_prototypes = self._normalize_prototypes(task_prototypes)
            if use_global_memory and self.use_global_memory:
                task_prototypes = self._fuse_global_memory(task_prototypes)
            query_features = self.prototype_generator.feature_encoder(query_data)
            logits = self._classify_with_prototypes(query_features, task_prototypes)
            predictions = torch.argmax(logits, dim=1)
            result = {
                'logits': logits,
                'predictions': predictions
            }
            if query_labels is not None:
                accuracy = self._compute_accuracy(logits, query_labels)
                unique_labels = torch.unique(query_labels)
                per_class_acc = {}
                for label in unique_labels:
                    mask = (query_labels == label)
                    if mask.sum() > 0:
                        class_acc = (predictions[mask] == query_labels[mask]).float().mean().item()
                        per_class_acc[label.item()] = class_acc
                probs = F.softmax(logits, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                result.update({
                    'accuracy': accuracy,
                    'per_class_accuracy': per_class_acc,
                    'confidence': {
                        'mean': max_probs.mean().item(),
                        'std': max_probs.std().item()
                    }
                })
        return result
    def _stabilize_input(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=self.input_clip_value, neginf=-self.input_clip_value)
        x = torch.clamp(x, min=-self.input_clip_value, max=self.input_clip_value)
        if self.config.get('normalize_input', True):
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + self.epsilon
            x = (x - mean) / std
        return x
    def _assess_adaptation_quality(self, losses: list) -> float:
        if not losses or len(losses) < 2:
            return 0.0
        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_reduction = initial_loss - final_loss
        reduction_rate = loss_reduction / (initial_loss + self.epsilon)
        loss_reduction_norm = max(0.0, min(1.0, reduction_rate))
        final_loss_norm = 1.0 / (1.0 + final_loss)
        quality_score = 0.6 * loss_reduction_norm + 0.4 * final_loss_norm
        return quality_score
    def _fuse_global_memory(self, task_prototypes: dict) -> dict:
        if not self.use_global_memory:
            return task_prototypes
        non_tensor_classes = []
        for cid, proto in task_prototypes.items():
            if not torch.is_tensor(proto):
                non_tensor_classes.append((cid, type(proto).__name__))
        if non_tensor_classes:
            logger.warning(
                f"  {len(non_tensor_classes)}  Tensor \n"
                f"   : {[cid for cid, _ in non_tensor_classes]}\n"
                f"   ..."
            )
            try:
                task_prototypes = self._normalize_prototypes(task_prototypes)
                logger.info(" ")
            except Exception as e:
                logger.error(f" : {e}")
                task_prototypes = {
                    cid: proto for cid, proto in task_prototypes.items()
                    if torch.is_tensor(proto)
                }
                if not task_prototypes:
                    raise ValueError(" ")
        fused_prototypes = {}
        for cid, main_proto in task_prototypes.items():
            if not torch.is_tensor(main_proto):
                logger.error(f"  {cid}  Tensor: {type(main_proto)}")
                continue
            if cid in self.global_prototype_memory and self.global_prototype_memory[cid] is not None:
                global_proto = self.global_prototype_memory[cid]
                if not torch.is_tensor(global_proto):
                    logger.warning(f"  {cid} ")
                    fused_prototypes[cid] = main_proto
                    self._memory_miss_count += 1
                    continue
                if global_proto.shape != main_proto.shape:
                    logger.warning(
                        f"  {cid} : "
                        f"global={global_proto.shape} vs main={main_proto.shape}"
                    )
                    fused_prototypes[cid] = main_proto
                    self._memory_miss_count += 1
                    continue
                fused_proto = (
                        self.alpha_fusion * main_proto +
                        (1 - self.alpha_fusion) * global_proto
                )
                fused_prototypes[cid] = fused_proto
                self._memory_hit_count += 1
            else:
                fused_prototypes[cid] = main_proto
                self._memory_miss_count += 1
        if len(fused_prototypes) == 0:
            logger.error(
                f" \n"
                f"   : {len(task_prototypes)} \n"
                f"    Tensor: {len(non_tensor_classes)} "
            )
            raise ValueError("")
        invalid_outputs = []
        for cid, proto in fused_prototypes.items():
            if not torch.is_tensor(proto):
                invalid_outputs.append(cid)
        if invalid_outputs:
            logger.error(f"  Tensor: {invalid_outputs}")
            fused_prototypes = {
                cid: proto for cid, proto in fused_prototypes.items()
                if torch.is_tensor(proto)
            }
        return fused_prototypes
    def update_global_memory(self, task_prototypes: dict):
        for cid, main_proto in task_prototypes.items():
            if not torch.is_tensor(main_proto):
                continue
            main_proto = main_proto.detach().clone()
            if cid not in self.global_prototype_memory or self.global_prototype_memory[cid] is None:
                self.global_prototype_memory[cid] = main_proto
                logger.debug(f"  {cid} ")
            else:
                old_proto = self.global_prototype_memory[cid]
                new_proto = (
                        self.beta_update * old_proto +
                        (1 - self.beta_update) * main_proto
                )
                self.global_prototype_memory[cid] = new_proto
            self.memory_update_count += 1
    def _classify_with_prototypes(
            self,
            features: torch.Tensor,
            prototypes: dict
    ) -> torch.Tensor:
        if not torch.is_tensor(features):
            raise TypeError(f"features  Tensor: {type(features)}")
        if not isinstance(prototypes, dict):
            raise TypeError(f"prototypes : {type(prototypes)}")
        if torch.isnan(features).any():
            logger.warning(" features  NaN,  0")
            features = torch.nan_to_num(features, nan=0.0)
        class_ids = sorted(prototypes.keys())
        prototype_list = []
        for cid in class_ids:
            proto = prototypes[cid]
            if not torch.is_tensor(proto):
                raise ValueError(f" {cid}  Tensor: {type(proto)}")
            if proto.dim() == 1:
                proto = proto.unsqueeze(0)
            elif proto.dim() > 2:
                proto = proto.view(-1, proto.size(-1))
                proto = proto.mean(dim=0, keepdim=True)
            if torch.isnan(proto).any():
                logger.warning(f"  {cid}  NaN,  0")
                proto = torch.nan_to_num(proto, nan=0.0)
            prototype_list.append(proto)
        prototype_matrix = torch.cat(prototype_list, dim=0)
        if features.size(-1) != prototype_matrix.size(-1):
            raise ValueError(
                f": features {features.shape} vs prototypes {prototype_matrix.shape}"
            )
        features_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        prototypes_norm = F.normalize(prototype_matrix, p=2, dim=1, eps=1e-8)
        similarities = torch.mm(features_norm, prototypes_norm.t())
        if torch.isnan(similarities).any():
            logger.warning(" similarities  NaN,  0")
            similarities = torch.nan_to_num(similarities, nan=0.0)
        logits = similarities * self.temperature
        logits = torch.clamp(logits, min=-100, max=100)
        return logits
    def _compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).float().sum()
        accuracy = correct / labels.size(0)
        return accuracy.item()
    def save_global_memory(self, save_path: str = None):
        if save_path is None:
            save_dir = self.config.get('checkpoint_dir', 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'global_prototype_memory.pt')
        memory_dict = {
            'prototypes': dict(self.global_prototype_memory),
            'update_count': self.memory_update_count,
            'episode_count': self._episode_count
        }
        torch.save(memory_dict, save_path)
        logger.info(f" : {save_path}")
    def load_global_memory(self, load_path: str = None):
        if load_path is None:
            save_dir = self.config.get('checkpoint_dir', 'checkpoints')
            load_path = os.path.join(save_dir, 'global_prototype_memory.pt')
        if not os.path.exists(load_path):
            logger.warning(f" : {load_path}")
            return
        memory_dict = torch.load(load_path, map_location=self.device, weights_only=False)
        self.global_prototype_memory = defaultdict(lambda: None, memory_dict['prototypes'])
        self.memory_update_count = memory_dict.get('update_count', 0)
        self._episode_count = memory_dict.get('episode_count', 0)
    def reset_global_memory(self):
        self.global_prototype_memory = defaultdict(lambda: None)
        self.memory_update_count = 0
        self._memory_hit_count = 0
        self._memory_miss_count = 0
        logger.info(" ")
    def _cache_first_linear(self):
        try:
            encoder = self.prototype_generator.feature_encoder
            for module in encoder.modules():
                if isinstance(module, nn.Linear):
                    self._first_linear_layer = module
                    logger.debug(f" : {module}")
                    break
            if self._first_linear_layer is None:
                logger.warning("  Linear ")
        except Exception as e:
            logger.warning(f" : {e}")
    def _init_weights_recursively(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    def _verify_initialization(self):
        zero_count = 0
        for name, param in self.named_parameters():
            if 'bias' in name and 'Linear' in name and param.numel() > 0:
                if abs(param.data.flatten()[0].item()) < 1e-6:
                    zero_count += 1
                    logger.warning(f" {name} ")
        if zero_count == 0:
            logger.info("  Linear bias ")
    def print_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  : {total_params:,}")
        print(f"  : {trainable_params:,}")
        print(f"  : {self.n_way}-way {self.k_shot}-shot")
        print(f"  : {len(self.global_prototype_memory)}")
        print(f"  Episode : {self._episode_count}")
        print(f"  : {self.temperature.item():.4f}")
