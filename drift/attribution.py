import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except ImportError:
    has_ipex = False
class MMD(nn.Module):
    def __init__(self, bandwidth: float = 1.0):
        super().__init__()
        self.bandwidth = bandwidth
    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = torch.sum(x**2, dim=1, keepdim=True)
        y_norm = torch.sum(y**2, dim=1, keepdim=True)
        dist = x_norm + y_norm.t() - 2 * torch.mm(x, y.t())
        return torch.exp(-dist / (2 * self.bandwidth**2))
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_xx = self.rbf_kernel(x, x)
        k_xy = self.rbf_kernel(x, y)
        k_yy = self.rbf_kernel(y, y)
        n = x.shape[0]
        m = y.shape[0]
        mmd = (torch.sum(k_xx) / (n * n) + torch.sum(k_yy) / (m * m) -
               2 * torch.sum(k_xy) / (n * m))
        return mmd
class DriftAttribution:
    def __init__(self, bandwidth: float = 1.0, device: torch.device = None):
        if device is not None:
            self.device = device
        elif has_ipex and torch.xpu.is_available():
            self.device = torch.device('xpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.mmd = MMD(bandwidth=bandwidth).to(self.device)
        logger.info(f" : {self.device}")
    def compute_drift_scores(self, source_features: torch.Tensor,
                           target_old_features: torch.Tensor,
                           target_new_features: torch.Tensor) -> dict:
        source_features = source_features.to(self.device)
        target_old_features = target_old_features.to(self.device)
        target_new_features = target_new_features.to(self.device)
        delta_domain = self.mmd(source_features, target_old_features)
        delta_temporal = self.mmd(target_old_features, target_new_features)
        return {
            'delta_domain': delta_domain.item(),
            'delta_temporal': delta_temporal.item()
        }
    def attribute_drift(self, delta_domain: torch.Tensor,
                       delta_temporal: torch.Tensor) -> dict:
        epsilon = 1e-8
        if isinstance(delta_domain, torch.Tensor):
            delta_domain = delta_domain.item()
        if isinstance(delta_temporal, torch.Tensor):
            delta_temporal = delta_temporal.item()
        max_delta = max(delta_domain, delta_temporal) + epsilon
        delta_domain_norm = delta_domain / max_delta
        delta_temporal_norm = delta_temporal / max_delta
        total_norm = delta_domain_norm + delta_temporal_norm + epsilon
        R_drift = delta_temporal_norm / total_norm
        w_domain = delta_domain_norm / total_norm
        w_temporal = delta_temporal_norm / total_norm
        drift_type = self._determine_drift_type(R_drift)
        return {
            'R_drift': R_drift,
            'w_domain': w_domain,
            'w_temporal': w_temporal,
            'drift_type': drift_type,
            'delta_domain_raw': delta_domain,
            'delta_temporal_raw': delta_temporal
        }
    def _determine_drift_type(self, R_drift: float) -> str:
        if R_drift < 0.3:
            return 'domain_shift'
        elif R_drift > 0.7:
            return 'temporal_drift'
        else:
            return 'coupled_drift'
    def __call__(self, source_features: torch.Tensor,
                target_old_features: torch.Tensor,
                target_new_features: torch.Tensor) -> dict:
        drift_scores = self.compute_drift_scores(
            source_features, target_old_features, target_new_features
        )
        attribution = self.attribute_drift(
            drift_scores['delta_domain'], drift_scores['delta_temporal']
        )
        result = {**drift_scores, **attribution}
        logger.info(f" : delta_domain={drift_scores['delta_domain']:.4f}, "
                   f"delta_temporal={drift_scores['delta_temporal']:.4f}, "
                   f"drift_type={attribution['drift_type']}")
        return result
class PrototypeDriftDetector:
    def __init__(self, bandwidth: float = 1.0, device: torch.device = None):
        if device is not None:
            self.device = device
        elif has_ipex and torch.xpu.is_available():
            self.device = torch.device('xpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.attributor = DriftAttribution(bandwidth=bandwidth, device=self.device)
    def detect_drift(self, source_prototypes: torch.Tensor,
                    target_old_prototypes: torch.Tensor,
                    target_new_prototypes: torch.Tensor) -> dict:
        return self.attributor(
            source_prototypes, target_old_prototypes, target_new_prototypes
        )
