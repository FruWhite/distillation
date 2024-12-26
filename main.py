import torch
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


logits_dist_losses = {"KD": losses.KLDiv,
                      "DIST": losses.DIST,
                      "DKD": losses.DKD,
                      "BASELINE": None}

optims = {"SGD": optim.SGD, "Adam": optim.Adam, "AdamW": optim.AdamW}

# dataset_names = tuple(s for s in INFO.keys() if "3d" not in s)
# ('pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist',
# 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist')
# dataset_names = (
# 'breastmnist', 'retinamnist', 'pneumoniamnist', 'dermamnist', 'bloodmnist', 'organamnist', 'organcmnist', 'organsmnist',
# 'pathmnist', 'octmnist', 'chestmnist', 'tissuemnist')


ResNet50testACC = {'breastmnist': 0.812, 'retinamnist': 0.528, 'pneumoniamnist': 0.854, 'dermamnist': 0.735,
                   'bloodmnist': 0.956, 'organamnist': 0.935, 'organcmnist': 0.905, 'organsmnist': 0.770,
'pathmnist': 0.911, 'octmnist': 0.762, 'chestmnist': 0.947, 'tissuemnist': 0.680}



def kd_main(config):
    assert config.train_type == 'logit_kd', "wrong training main func"
    class_num = config.class_num

    if not config.use_saved_teacher_logits:
        teacher = ViTTeacher(class_num).to(DEVICE)
        teacher.eval()
    else:
        teacher = None

    student = StudentModel(class_num).to(DEVICE)
    optimizer = optims[config.optim](student.parameters(), lr=config.lr)
    train_loader, val_loader, test_loader = load_medmnist(config, shuffle=False)

    if logits_dist_losses[config.loss_type] is not None:
        loss_fn = losses.LogitsDistillLoss(logits_dist_losses[config.loss_type](*config.loss_init_params))
        kd_train(student, teacher, train_loader, val_loader, optimizer, loss_fn, config)
    else:
        base_train(student, train_loader, val_loader, optimizer, config)

    torch.save(student.state_dict(),
               f"students/{config.dataset_name}_stu_fromResNet50_{config.loss_type}_{config.epochs}ep_{config.batchsize}bs.pth")
    test_acc = evaluate(student, test_loader, config)
    print(f"Final Eval for {config.dataset_name}_{config.loss_type}, Test Acc: {test_acc}\n")
    return test_acc




def main_logits(config, datasets, losses):
    # test for all 2d medmnist datasets and all logits_based loss
    results = []

    for dataset in tqdm(datasets):
        row = []
        config.dataset_name = dataset
        config.class_num = len(INFO[dataset]['label'])

        for loss in tqdm(losses):
            print(f"{dataset}, {loss}")
            config.loss_type = loss
            test_acc = kd_main(config)
            row.append(test_acc)
        results.append(row)
        df = pd.DataFrame(results, columns=losses)
        df.to_csv(f"result/ckpt_result_{dataset}.csv", sep=',', index=True, header=True)
    df = pd.DataFrame(results, columns=losses, index=datasets)
    return df


if __name__ == "__main__":
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    wandb.login(key="0124466f811cc017af995c30c7924f5e40bfda31")

    # for MacOS local run:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("use mps as DEVICE")
    else:
        print("MPS device not found, use CUDA.")
        assert torch.cuda.is_available(), "CUDA is not available"
        DEVICE = "cuda"

    # problem Make sure that the channel dimension of the pixel values match with the one set in the configuration. Expected 3 but got 1.

    # dataset_name = 'bloodmnist'
    # dataset_name = "pathmnist"
    # dataset_name = 'chestmnist'
    # dataset_name = 'dermamnist'
    # dataset_name = 'octmnist'
    # dataset_name = 'pneumoniamnist'
    # dataset_name = 'retinamnist'
    # dataset_name = 'breastmnist'
    # dataset_name = 'tissuemnist'
    dataset_name = 'organamnist'
    # dataset_name = 'organsmnist'
    # dataset_name = 'organcmnist'

    config = Namespace(
        project_name="cv-pj-new",
        batchsize=256, # MacOS: 32, cuda: 4090d(24G): 256[ok]
        lr=1e-3,
        optim='Adam',
        epochs=10,
        ckpt_path='checkpoint.pt',
        dataset_name=dataset_name,
        class_num=len(INFO[dataset_name]['label']),
        train_type="logit_kd",
        loss_type="KD",
        loss_init_params=(),
        use_saved_teacher_logits=True,
        teacher_test_acc = ResNet50testACC[dataset_name],
        device = DEVICE
    )
    # # show_data(load_medmnist(config)[0])
    # save_teacher_logits(config)
    # train_loader, val_loader, test_loader = load_medmnist(config)
    # show_data(val_loader)

    # kd_main(config)
    # i = 0 # 0, 1, 2, 3, 4, 5
    # print(dataset_names)

    dataset_names = (
        'breastmnist', 'retinamnist', 'pneumoniamnist', 'dermamnist', 'bloodmnist', 'organamnist', 'organcmnist',
        'organsmnist',
        'pathmnist', 'octmnist', 'chestmnist', 'tissuemnist')

    set = []
    set.append(('breastmnist', 'retinamnist', 'pneumoniamnist', 'dermamnist',
            'bloodmnist', 'organcmnist', 'organsmnist'))
    set.append(('organamnist'))
    set.append(('pathmnist'))
    set.append(('octmnist'))
    set.append(('chestmnist'))

    i = 0 # 0, 1, 2, 3, 4
    df = main_logits(config, set[i], logits_dist_losses.keys())

    print(df)
    df.to_csv(f"result/final_{i}.csv", sep=',', index=True, header=True)


