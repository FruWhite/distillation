import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import os
import pandas as pd

import wandb
import random
from argparse import Namespace

import losses
from dataload import load_medmnist, show_data, save_teacher_logits
from models import ViTTeacher, StudentModel
from train import *
from medmnist import INFO
import losses

DEVICE = "cuda"
logits_dist_losses = {"KD": losses.KLDiv,
                          "DIST": losses.DIST,
                          "DKD": losses.DKD, }

optims = {"SGD": optim.SGD, "Adam": optim.Adam, "AdamW": optim.AdamW}

dataset_names = tuple(s for s in INFO.keys() if "3d" not in s)
"""
('pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist',
 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist')
"""


def kd_main(config):
    assert config.train_type == 'logit_kd', "wrong training main func"
    class_num = config.class_num

    if not config.teacher_logits_available:
        teacher = ViTTeacher(class_num).to(DEVICE)
    else:
        teacher = None

    student = StudentModel(class_num).to(DEVICE)
    optimizer = optims[config.optim](student.parameters(), lr=config.lr)
    loss_fn = losses.LogitsDistillLoss(logits_dist_losses[config.loss_type](*config.loss_init_params))

    train_loader, val_loader, test_loader = load_medmnist(config)
    kd_train(student, teacher, train_loader, val_loader, optimizer, loss_fn, config)
    torch.save(student.state_dict(),
               f"students/{config.dataset_name}_stu_fromViT_{config.loss_type}_{config.epochs}ep_{config.batchsize}bs.pth")
    test_acc = evaluate(student, test_loader)
    print(f"Final Eval for {config.dataset_name}_{config.loss_type}, Test Acc: {test_acc}\n")
    return test_acc




def main_logits(config, datasets, losses):
    # test for all 2d medmnist datasets and all logits_based loss
    results = []

    for dataset in tqdm(datasets):
        row = []
        config.dataset_name = dataset
        for loss in tqdm(losses):
            config.loss_type = loss
            test_acc = kd_main(config)
            row.append(test_acc)
        results.append(row)

    df = pd.DataFrame(results, columns=losses, index=datasets)
    return df


if __name__ == "__main__":
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    wandb.login()


# problem Make sure that the channel dimension of the pixel values match with the one set in the configuration. Expected 3 but got 1.

    # dataset_name = 'bloodmnist'
    # dataset_name = "pathmnist"
    # dataset_name = 'chestmnist'
    # dataset_name = 'dermamnist'
    # dataset_name = 'octmnist'
    # dataset_name = 'pneumoniamnist'
    # dataset_name = 'retinamnist'
    # dataset_name = 'breastmnist'
    dataset_name = 'tissuemnist'
    # dataset_name = 'organamnist'
    # dataset_name = 'organsmnist'
    # dataset_name = 'organcmnist'
    config = Namespace(
        project_name="cv-pj",
        batchsize=32,
        lr=1e-3,
        optim='Adam',
        epochs=10,
        ckpt_path='checkpoint.pt',
        dataset_name = dataset_name,
        class_num=len(INFO[dataset_name]['label']),
        train_type = "logit_kd",
        loss_type = "KD",
        loss_init_params=(),
        teacher_logits_available = True,
    )
    # show_data(load_medmnist(config)[0])
    save_teacher_logits(config)
    # train_loader, val_loader, test_loader = load_medmnist(config)
    # show_data(val_loader)

    # kd_main(config)

    # for v in logits_dist_losses.values():
    #     config.loss_type = v
    #     kd_main(config)













