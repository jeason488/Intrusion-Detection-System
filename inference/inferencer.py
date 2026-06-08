import torch
import torch.nn.functional as F
import os
import logging
import json
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class Inferencer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        self.model.eval()
    def _build_model(self):
        from models.dhpen_network import DHPENNetwork
        return DHPENNetwork(self.config).to(self.device)
    def load_model(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            logger.error(f": {checkpoint_path}")
            return False
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
            logger.info(f": {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f": {e}")
            return False
    def predict(self, features, support_data=None, support_labels=None):
        self.model.eval()
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32).to(self.device)
        else:
            features = features.to(self.device)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if support_data is None:
            n_way, feature_dim = self.config.get('n_way', 2), self.config.get('input_dim', 59)
            support_x = torch.randn(n_way * 5, feature_dim).to(self.device) * 0.3
            support_y = torch.arange(n_way).repeat(5).to(self.device)
        else:
            support_x = torch.tensor(support_data, dtype=torch.float32).to(self.device) if not isinstance(support_data, torch.Tensor) else support_data
            support_y = torch.tensor(support_labels, dtype=torch.long).to(self.device) if not isinstance(support_labels, torch.Tensor) else support_labels
        with torch.no_grad():
            result = self.model.meta_test_step(support_x, support_y, features)
        predictions = result.get('predictions', torch.tensor([0]))
        logits = result.get('logits', torch.tensor([[0.5, 0.5]]))
        probs = F.softmax(logits, dim=1)
        confidence = float(probs[0, predictions[0]]) if predictions.numel() > 0 else 0.5
        return {
            'predictions': predictions.cpu().numpy().tolist(),
            'logits': logits.cpu().numpy().tolist(),
            'confidence': confidence,
            'probabilities': probs.cpu().numpy().tolist()
        }
    def batch_predict(self, features_list, support_data=None, support_labels=None):
        return [self.predict(f, support_data, support_labels) for f in features_list]
    def test(self, checkpoint_path=None):
        if checkpoint_path:
            self.load_model(checkpoint_path)
        logger.info("")
        total_accuracy = 0.0
        results = []
        for episode in range(10):
            support_x, support_y, query_x, query_y = self._generate_test_episode()
            result = self.model.meta_test_step(support_x, support_y, query_x, query_y)
            accuracy = result.get('accuracy', 0.0)
            total_accuracy += accuracy
            results.append({'episode': episode + 1, 'accuracy': accuracy})
            logger.info(f"Episode {episode + 1}:  = {accuracy:.4f}")
        avg_accuracy = total_accuracy / 10
        test_result = {'avg_accuracy': avg_accuracy, 'n_episodes': 10, 'results': results, 'timestamp': datetime.now().isoformat()}
        output_dir = self.config.get('training', {}).get('checkpoint_dir', './checkpoints')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'test_result.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=2, ensure_ascii=False)
        logger.info(f": {avg_accuracy:.4f}")
        logger.info(f": {path}")
        return test_result
    def _generate_test_episode(self):
        n_way = self.config.get('n_way', 5)
        k_shot = self.config.get('k_shot', 5)
        n_query = self.config.get('n_query', 15)
        feature_dim = self.config.get('input_dim', 59)
        support_x = torch.randn(n_way * k_shot, feature_dim).to(self.device) * 0.3
        support_y = torch.arange(n_way).repeat(k_shot).to(self.device)
        query_x = torch.randn(n_way * n_query, feature_dim).to(self.device) * 0.3
        query_y = torch.arange(n_way).repeat(n_query).to(self.device)
        for i in range(n_way):
            class_mask = support_y == i
            support_x[class_mask] += i * 0.5
            class_mask_q = query_y == i
            query_x[class_mask_q] += i * 0.5
        return support_x, support_y, query_x, query_y
    def analyze_sample(self, features):
        result = self.predict(features)
        analysis = {
            'input_features': features if isinstance(features, list) else features.tolist(),
            'prediction': result['predictions'][0],
            'is_anomaly': result['predictions'][0] == 1,
            'confidence': result['confidence'],
            'probabilities': result['probabilities'][0],
            'analysis': {
                'confidence_level': '' if result['confidence'] > 0.8 else '' if result['confidence'] > 0.5 else '',
                'interpretation': '' if result['predictions'][0] == 1 else '',
                'recommendation': '' if result['predictions'][0] == 1 else ''
            },
            'timestamp': datetime.now().isoformat()
        }
        return analysis
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            'model_name': 'DHPENNetwork',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'config': self.config,
            'is_training': self.model.training
        }
