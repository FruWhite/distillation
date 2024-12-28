import torch.nn as nn
import torch.nn.functional as F


class CTKD(nn.Module):
    def __init__(self, temperature=1.0, max_temp=5.0, total_steps=1870, alpha=0.8):
        super().__init__()
        self.temperature = temperature
        self.max_temp = max_temp
        self.alpha = alpha
        self.total_steps = total_steps
        self.current_step = 0
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def update_temperature(self):
        """
        动态更新温度，从 init_temp 线性增加到 max_temp。
        """
        progress = min(self.current_step / self.total_steps, 1.0)
        self.temperature = self.temperature + (self.max_temp - self.temperature) * progress

    def forward(self, student_logits, teacher_logits, true_labels):
        # 更新当前温度
        self.update_temperature()
        self.current_step += 1
        # Soft targets (来自教师网络的预测)
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        # KL 散度损失
        distillation_loss = self.kl_div_loss(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            soft_targets
        )
        true_labels = true_labels.flatten()
        classification_loss = self.ce_loss(student_logits, true_labels)
        # 混合损失
        return self.alpha * distillation_loss + (1 - self.alpha) * classification_loss



