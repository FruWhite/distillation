import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
import torch.optim as optim
from transformers import AutoModelForImageClassification
# 教师模型


class ViTTeacher(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model.classifier = nn.Linear(self.model.config.hidden_size, class_num)

    def forward(self, x):
        return self.model(x).logits


class StudentModel(nn.Module):
    def __init__(self, class_num):
        super(StudentModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier[1] = nn.Linear(1280, class_num)

    def forward(self, x):
        return self.mobilenet(x)