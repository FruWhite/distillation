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
logits_dist_losses = {"KLDiv": losses.KLDiv,
                          "DIST": losses.DIST,
                          "DKD": losses.DKD, }

optims = {"SGD": optim.SGD, "Adam": optim.Adam, "AdamW": optim.AdamW}

def kd_main(config):
    assert config.train_type == 'logit_kd', "wrong training main func"
    class_num = config.class_num
    teacher = ViTTeacher(class_num).to(DEVICE)
    student = StudentModel(class_num).to(DEVICE)
    optimizer = optims[config.optim](student.parameters(), lr=config.lr)
    loss_fn = losses.LogitsDistillLoss(config.loss_type(*config.loss_init_params))
    kd_train(student, teacher, train_loader, val_loader, optimizer, loss_fn, config)
    torch.save(student.state_dict(), "student_model_vit_kd_5ep.pth")
    print(f"Final Eval, Test Acc: {evaluate(student, test_loader)}\n")





if __name__ == "__main__":
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    wandb.login()




    dataset_name = 'bloodmnist'
    config = Namespace(
        project_name="cv-pj",
        batchsize=32,
        lr=1e-3,
        optim='Adam',
        epochs=5,
        ckpt_path='checkpoint.pt',
        dataset_name = dataset_name,
        class_num=len(INFO[dataset_name]['label']),
        train_type = "logit_kd",
        loss_type = None,
        loss_init_params=(),
    )

    train_loader, val_loader, test_loader = load_medmnist(config)
    # show_data(val_loader)

    for v in logits_dist_losses.values():
        config.loss_type = v
        kd_main(config)













