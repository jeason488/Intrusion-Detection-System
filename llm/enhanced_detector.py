import torch
import torch.nn as nn
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
class EnhancedDetector(nn.Module):
    def __init__(self, config: dict):
        super(EnhancedDetector, self).__init__()
        self.config = config
        self.feature_dim = config.get('input_dim', 78)
        self.llm_embedding_dim = config.get('llm_embedding_dim', 768)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.feature_dim + self.llm_embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2)
        )
    def forward(self, features, llm_embeddings):
        fused_features = torch.cat([features, llm_embeddings], dim=1)
        logits = self.fusion_layer(fused_features)
        return logits
    def detect(self, features, llm_interface):
        feature_text = self._features_to_text(features)
        llm_embeddings = llm_interface.encode(feature_text)
        logits = self.forward(features, llm_embeddings)
        predictions = torch.argmax(logits, dim=1)
        confidence = torch.softmax(logits, dim=1).max(dim=1)[0]
        return {
            'logits': logits,
            'predictions': predictions,
            'confidence': confidence
        }
    def _features_to_text(self, features):
        text = ": "
        text += f": {features[0].item():.2f}, "
        text += f": {features[4].item():.0f}, "
        text += f": {features[5].item():.0f}, "
        text += f": {features[6].item():.0f}, "
        text += f": {features[7].item():.0f}"
        return text
