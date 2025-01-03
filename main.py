import torch
import os
import pandas as pd
from torch import optim
import wandb
import random
from argparse import Namespace

import losses
from dataload import load_medmnist, show_data, save_teacher_logits
from models import ViTTeacher, StudentModel
from train import *
from medmnist import INFO
import losses
from sklearn.metrics import f1_score


logits_dist_losses = {"KD": losses.KLDiv,
                      "DIST": losses.DIST,
                      "DKD": losses.DKD,
                      "BASELINE": None}

optims = {"SGD": optim.SGD, "Adam": optim.Adam, "AdamW": optim.AdamW}

teachers = ("resnet50", "vit")
# dataset_names = tuple(s for s in INFO.keys() if "3d" not in s)


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


def f1score_main(config):
    assert config.train_type == 'f1score', "wrong training main func"
    class_num = config.class_num

    train_loader, val_loader, test_loader = load_medmnist(config)

    # 加载已经训练好的模型
    model_path = './students/retinamnist_stu_fromResNet50_KD_10ep_256bs.pth'
    baseline_stu_model = StudentModel(class_num).to(DEVICE)
    baseline_stu_model.load_state_dict(torch.load(model_path))
    baseline_stu_model.eval()  # 切换到评估模式

    f1_score_value = evaluate_f1_score(baseline_stu_model, test_loader, config)
    print(f"F1 Score: {f1_score_value:.4f}")
    return f1_score_value


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

    # for MacOS local run, if use cuda, ignore it
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("use mps as DEVICE")
    else:
        print("MPS device not found, use CUDA.")
        assert torch.cuda.is_available(), "CUDA is not available"
        DEVICE = "cuda"

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
        batchsize=256,
        # MacOS(local), vit: 32,
        # resnet:
        # 4090d(24G): 256[ok, 2.6~3.5s/it, full]
        # A800(80G): 1024[ok, 4.9~5.2s/it, full]
        # H20(80G): 1024[ok, 7.6~8.3s/it, full]
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
        teacher="resnet50",
        teacher_test_acc=ResNet50testACC[dataset_name],
        device=DEVICE
    )
    # # show_data(load_medmnist(config)[0])
    # save_teacher_logits(config)
    # train_loader, val_loader, test_loader = load_medmnist(config)
    # show_data(val_loader)

    # kd_main(config)

    dataset_names = (
        'breastmnist', 'retinamnist', 'pneumoniamnist', 'dermamnist', 'bloodmnist', 'organamnist', 'organcmnist',
        'organsmnist',
        'pathmnist', 'octmnist', 'chestmnist', 'tissuemnist')

    set = []

    # set.append(('breastmnist', 'retinamnist', 'pneumoniamnist', 'dermamnist',
    #         'bloodmnist', 'organcmnist', 'organsmnist'))
    set.append(('bloodmnist', 'organcmnist', 'organsmnist'))
    set.append(('organamnist',))
    set.append(('pathmnist',))
    set.append(('octmnist',))
    set.append(('chestmnist',))

    i = 0  # 0, 1, 2, 3, 4
    df = main_logits(config, set[i], logits_dist_losses.keys())

    print(df)
    df.to_csv(f"result/final_{i}.csv", sep=',', index=True, header=True)
