import torch
import torch.optim as optim
import os
import logging
import json
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = config.get('training', {}).get('epochs', 100)
        self.learning_rate = config.get('training', {}).get('learning_rate', 0.001)
        self.weight_decay = config.get('training', {}).get('weight_decay', 0.0001)
        self.checkpoint_dir = config.get('training', {}).get('checkpoint_dir', './checkpoints')
        self.log_interval = config.get('training', {}).get('log_interval', 10)
        self.save_interval = config.get('training', {}).get('save_interval', 50)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.best_accuracy = 0.0
        self.train_history = []
        self.val_history = []
    def _build_model(self):
        from models.dhpen_network import DHPENNetwork
        return DHPENNetwork(self.config).to(self.device)
    def _build_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    def _build_scheduler(self):
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
    def _generate_episode(self, n_way=5, k_shot=5, n_query=15, feature_dim=59):
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
    def train(self):
        logger.info("")
        self.model.train()
        n_episodes = 10
        for epoch in range(self.epochs):
            epoch_loss, epoch_accuracy = 0.0, 0.0
            for episode in range(n_episodes):
                support_x, support_y, query_x, query_y = self._generate_episode(
                    n_way=self.config.get('n_way', 5),
                    k_shot=self.config.get('k_shot', 5),
                    n_query=self.config.get('n_query', 15),
                    feature_dim=self.config.get('input_dim', 59)
                )
                metrics = self.model.meta_train_step(support_x, support_y, query_x, query_y, self.optimizer)
                epoch_loss += metrics['meta_loss']
                epoch_accuracy += metrics['accuracy']
            epoch_loss /= n_episodes
            epoch_accuracy /= n_episodes
            self.scheduler.step()
            self.train_history.append({'epoch': epoch + 1, 'loss': epoch_loss, 'accuracy': epoch_accuracy, 'learning_rate': self.scheduler.get_last_lr()[0]})
            if (epoch + 1) % self.log_interval == 0:
                logger.info(f"Epoch [{epoch + 1}/{self.epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f} | LR: {self.scheduler.get_last_lr()[0]:.6f}")
            if epoch_accuracy > self.best_accuracy:
                self.best_accuracy = epoch_accuracy
                self._save_checkpoint(epoch + 1, is_best=True)
                logger.info(f": {self.best_accuracy:.4f}")
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch + 1)
            if (epoch + 1) % 20 == 0:
                self._validate()
        self._save_training_history()
        logger.info("")
    def _validate(self):
        self.model.eval()
        total_accuracy = 0.0
        with torch.no_grad():
            for _ in range(5):
                support_x, support_y, query_x, query_y = self._generate_episode(
                    n_way=self.config.get('n_way', 5), k_shot=self.config.get('k_shot', 5),
                    n_query=self.config.get('n_query', 15), feature_dim=self.config.get('input_dim', 59)
                )
                total_accuracy += self.model.meta_test_step(support_x, support_y, query_x, query_y).get('accuracy', 0.0)
        avg_accuracy = total_accuracy / 5
        self.val_history.append({'epoch': len(self.train_history), 'accuracy': avg_accuracy})
        logger.info(f": {avg_accuracy:.4f}")
        self.model.train()
    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': self.config
        }
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        logger.info(f": {path}")
    def _save_training_history(self):
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_accuracy': self.best_accuracy,
            'config': self.config
        }
        path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        logger.info(f": {path}")
    def evaluate(self, checkpoint_path=None):
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        self.model.eval()
        n_episodes = 10
        total_accuracy = 0.0
        results = []
        with torch.no_grad():
            for episode in range(n_episodes):
                support_x, support_y, query_x, query_y = self._generate_episode(
                    n_way=self.config.get('n_way', 5),
                    k_shot=self.config.get('k_shot', 5),
                    n_query=self.config.get('n_query', 15),
                    feature_dim=self.config.get('input_dim', 59)
                )
                result = self.model.meta_test_step(support_x, support_y, query_x, query_y)
                total_accuracy += result.get('accuracy', 0.0)
                results.append({
                    'episode': episode + 1,
                    'accuracy': result.get('accuracy', 0.0),
                    'per_class_accuracy': result.get('per_class_accuracy', {}),
                    'confidence': result.get('confidence', {})
                })
        avg_accuracy = total_accuracy / n_episodes
        evaluation_result = {
            'avg_accuracy': avg_accuracy,
            'n_episodes': n_episodes,
            'results': results,
            'config': self.config
        }
        path = os.path.join(self.checkpoint_dir, 'evaluation_result.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
        logger.info(f": {avg_accuracy:.4f}")
        return evaluation_result
    def _load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            logger.error(f": {checkpoint_path}")
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        logger.info(f": {checkpoint_path}")
