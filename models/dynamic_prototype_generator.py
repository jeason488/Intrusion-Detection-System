import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
class DynamicPrototypeGenerator(nn.Module):
    def __init__(self, config: dict):
        super(DynamicPrototypeGenerator, self).__init__()
        self.config = config
        self.feature_dim = config.get('input_dim', 10)
        self.prototype_dim = config.get('prototype_dim', 128)
        self.feature_encoder = self._build_feature_encoder()
        self.prototype_constructor = self._build_prototype_constructor()
        self.evolution_engine = self._build_evolution_engine()
    def _build_feature_encoder(self):
        layers = []
        input_dim = self.feature_dim
        hidden_dims = [256, 512, 256, self.prototype_dim]
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Dropout(0.1))
            input_dim = dim
        return nn.Sequential(*layers)
    def _build_prototype_constructor(self):
        return nn.Sequential(
            nn.Linear(self.prototype_dim, self.prototype_dim),
            nn.GELU(),
            nn.Linear(self.prototype_dim, self.prototype_dim)
        )
    def _build_evolution_engine(self):
        return nn.Sequential(
            nn.Linear(self.prototype_dim, self.prototype_dim),
            nn.GELU(),
            nn.Linear(self.prototype_dim, self.prototype_dim)
        )
    def forward(self, support_data, support_labels, query_data, query_labels):
        support_features = self.feature_encoder(support_data)
        query_features = self.feature_encoder(query_data)
        initial_prototypes = self._generate_initial_prototypes(support_features, support_labels)
        evolved_prototypes = self._evolve_prototypes(initial_prototypes, support_features, support_labels)
        return {
            'support_features': support_features,
            'query_features': query_features,
            'initial_prototypes': initial_prototypes,
            'evolved_prototypes': evolved_prototypes
        }
    def _generate_initial_prototypes(self, support_features, support_labels):
        prototypes = {}
        unique_labels = torch.unique(support_labels)
        for label in unique_labels:
            class_features = support_features[support_labels == label]
            if len(class_features) > 0:
                prototype = class_features.mean(dim=0)
                prototypes[label.item()] = prototype
        return prototypes
    def _evolve_prototypes(self, initial_prototypes, support_features, support_labels):
        evolved_prototypes = {}
        for label, prototype in initial_prototypes.items():
            class_features = support_features[support_labels == label]
            if len(class_features) > 0:
                evolved_proto = self.evolution_engine(prototype)
                class_mean = class_features.mean(dim=0)
                class_std = class_features.std(dim=0)
                evolved_proto = 0.6 * evolved_proto + 0.3 * class_mean + 0.1 * class_std
                evolved_prototypes[label] = evolved_proto
        return evolved_prototypes
