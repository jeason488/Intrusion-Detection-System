import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
class KnowledgeDistillation:
    def __init__(self, config: dict):
        self.config = config
        self.temperature = config.get('distillation_temperature', 1.0)
        self.alpha = config.get('distillation_alpha', 0.7)
    def compute_distillation_loss(self, student_logits, teacher_logits, labels):
        soft_labels = F.softmax(teacher_logits / self.temperature, dim=1)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(student_soft, soft_labels, reduction='batchmean') * (self.temperature ** 2)
        hard_loss = F.cross_entropy(student_logits, labels)
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        return total_loss
    def distill(self, student_model, teacher_model, dataloader, optimizer, epochs=10):
        device = next(student_model.parameters()).device
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            for batch in dataloader:
                data, labels = batch
                data = data.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    teacher_logits = teacher_model(data)
                student_logits = student_model(data)
                loss = self.compute_distillation_loss(student_logits, teacher_logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = student_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            avg_loss = total_loss / len(dataloader)
            accuracy = 100. * correct / total
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
