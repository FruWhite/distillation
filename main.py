import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import os

import wandb
import random
from argparse import Namespace

import losses
from dataload import load_medmnist, show_data
from models import ViTTeacher, StudentModel
from train import *
from medmnist import INFO
import losses

DEVICE = "cuda"
config = Namespace(
    project_name = "cv-pj",
    batchsize = 32,
    lr = 1e-3,
    optim_type = 'Adam',
    epochs = 5,
    ckpt_path = 'checkpoint.pt',
)

if __name__ == "__main__":
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    wandb.login()

    train_loader, val_loader, test_loader = load_medmnist(config)
    class_num = len(INFO['pathmnist']['label'])
    # show_data(val_loader)
    teacher = ViTTeacher(class_num).to(DEVICE)
    student = StudentModel(class_num).to(DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=config.lr)
    loss_fn = losses.LogitsDistillLoss(losses.KLDiv())
    kd_train(student, teacher, train_loader, val_loader, optimizer, loss_fn, config)
    torch.save(student.state_dict(), "student_model_vit_kd_5ep.pth")
    print(evaluate(student, test_loader))











