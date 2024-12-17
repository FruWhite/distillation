import torch.nn as nn
import torch.nn.functional as F

class LogitsDistillLoss(nn.Module):
    def __init__(self, logitsloss, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.logitloss=logitsloss
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, true_labels):
        true_labels=true_labels.flatten()
        # 混合损失
        return self.alpha * self.logitloss(student_logits, teacher_logits) \
            + (1 - self.alpha) * self.ce_loss(student_logits, true_labels)