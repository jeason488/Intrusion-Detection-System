import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except ImportError:
    has_ipex = False
class AdaptationController:
    def __init__(self, lambda_: float = 0.5, smoothing: float = 1e-10, device: torch.device = None,
                 confidence_threshold: float = 0.5):
        self.lambda_ = lambda_
        self.smoothing = smoothing
        if device is not None:
            self.device = device
        elif has_ipex and torch.xpu.is_available():
            self.device = torch.device('xpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.confidence_threshold = confidence_threshold
        logger.info(f" : {self.device}")
    def compute_alignment_prototype(self, prototypes_old: torch.Tensor, source_prototypes: torch.Tensor) -> torch.Tensor:
        alignment_prototype = prototypes_old + (source_prototypes - prototypes_old) * 0.5
        return alignment_prototype
    def compute_evolution_prototype(self, new_features: torch.Tensor, new_labels: torch.Tensor, num_classes: int,
                                   prototypes: torch.Tensor, confidence_threshold: float = 0.5) -> torch.Tensor:
        features_norm = F.normalize(new_features, dim=1)
        prototypes_norm = F.normalize(prototypes, dim=1)
        similarities = torch.mm(features_norm, prototypes_norm.t())
        probabilities = F.softmax(similarities, dim=1)
        max_prob, _ = probabilities.max(dim=1)
        confident_mask = max_prob >= confidence_threshold
        confident_features = new_features[confident_mask]
        confident_labels = new_labels[confident_mask]
        evolution_prototype = torch.zeros(num_classes, new_features.shape[1], device=self.device)
        for c in range(num_classes):
            mask = (confident_labels == c)
            if mask.sum() > 0:
                class_features = confident_features[mask]
                evolution_prototype[c] = class_features.mean(dim=0)
            else:
                evolution_prototype[c] = torch.zeros(new_features.shape[1], device=self.device)
        return evolution_prototype
    def compute_weights(self, delta_domain: torch.Tensor, delta_temporal: torch.Tensor) -> tuple:
        if isinstance(delta_domain, torch.Tensor):
            delta_domain = delta_domain.item()
        if isinstance(delta_temporal, torch.Tensor):
            delta_temporal = delta_temporal.item()
        total_drift = delta_domain + delta_temporal + self.smoothing
        w_domain = delta_domain / total_drift
        w_temporal = delta_temporal / total_drift
        w_retention = self.lambda_ / (w_domain + w_temporal + self.lambda_ + self.smoothing)
        total_weight = w_domain + w_temporal + w_retention
        w_d = w_domain / total_weight
        w_t = w_temporal / total_weight
        w_r = w_retention / total_weight
        return w_d, w_t, w_r
    def update_prototypes(self, prototypes_old: torch.Tensor, new_features: torch.Tensor,
                         new_labels: torch.Tensor, source_prototypes: torch.Tensor,
                         delta_domain: torch.Tensor, delta_temporal: torch.Tensor) -> torch.Tensor:
        prototypes_old = prototypes_old.to(self.device)
        new_features = new_features.to(self.device)
        new_labels = new_labels.to(self.device)
        source_prototypes = source_prototypes.to(self.device)
        p_align = self.compute_alignment_prototype(prototypes_old, source_prototypes)
        num_classes = prototypes_old.shape[0]
        p_evolve = self.compute_evolution_prototype(new_features, new_labels, num_classes,
                                                  prototypes_old, self.confidence_threshold)
        p_old = prototypes_old
        w_d, w_t, w_r = self.compute_weights(delta_domain, delta_temporal)
        p_new = w_d * p_align + w_t * p_evolve + w_r * p_old
        alpha = self.lambda_
        p_final = alpha * p_old + (1 - alpha) * p_new
        logger.info(f" : w_d={w_d:.4f}, w_t={w_t:.4f}, w_r={w_r:.4f}")
        return p_final
    def __call__(self, prototypes_old: torch.Tensor, new_features: torch.Tensor,
                 new_labels: torch.Tensor, source_prototypes: torch.Tensor,
                 delta_domain: torch.Tensor, delta_temporal: torch.Tensor) -> torch.Tensor:
        return self.update_prototypes(
            prototypes_old, new_features, new_labels, source_prototypes,
            delta_domain, delta_temporal
        )
