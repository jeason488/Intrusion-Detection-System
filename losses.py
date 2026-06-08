import torch
import torch.nn as nn
import torch.nn.functional as F
class EnhancedMetaLoss(nn.Module):
    def __init__(self, config: dict):
        super(EnhancedMetaLoss, self).__init__()
        self.config = config
        self.alpha = config.get('alpha', 0.5)
        self.beta = config.get('beta', 0.3)
        self.gamma = config.get('gamma', 0.2)
    def forward(self, logits, labels, info):
        classification_loss = F.cross_entropy(logits, labels)
        prototype_loss = 0.0
        if 'evolved_prototypes' in info:
            prototypes = info['evolved_prototypes']
            if prototypes:
                prototype_list = list(prototypes.values())
                if len(prototype_list) > 1:
                    prototype_matrix = torch.stack(prototype_list)
                    norm_prototypes = F.normalize(prototype_matrix, dim=1)
                    similarity_matrix = torch.mm(norm_prototypes, norm_prototypes.t())
                    mask = torch.eye(len(prototype_list), device=similarity_matrix.device).bool()
                    similarity_matrix = similarity_matrix.masked_fill(mask, 0)
                    avg_similarity = similarity_matrix.sum() / (len(prototype_list) * (len(prototype_list) - 1))
                    prototype_loss = avg_similarity
        regularization_loss = 0.0
        total_loss = (
            self.alpha * classification_loss +
            self.beta * prototype_loss +
            self.gamma * regularization_loss
        )
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss.item(),
            'prototype_loss': prototype_loss.item(),
            'regularization_loss': regularization_loss
        }
