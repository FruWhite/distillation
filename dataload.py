import medmnist
from medmnist import INFO
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from models import ViTTeacher
import os
import pandas as pd

DEVICE = "cuda"


def is_rgb(D):
    # Check if images are RGB or grayscale by looking at the number of channels in the first image
    dataset = D(split='train', download=True)
    sample_image, _ = dataset[0]
    return sample_image.size[0] == 3  # Check the number of channels (should be 3 for RGB)


def load_medmnist(config, shuffle=True):

    BATCH_SIZE = config.batchsize
    dataset_name = config.dataset_name
    if config.use_saved_teacher_logits:
        shuffle = False
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])

    if is_rgb(DataClass):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels (RGB)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])
    # 加载数据集
    train_dataset = DataClass(split='train', transform=transform, download=True)
    val_dataset = DataClass(split='val', transform=transform, download=True)
    test_dataset = DataClass(split='test', transform=transform, download=True)

    # 构建 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def show_data(loader):
    # 获取十个训练样本
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # 创建一个图像显示窗口
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    # 绘制图像并标注标签
    for i in range(10):
        ax = axes[i // 5, i % 5]
        image = images[i].numpy().transpose((1, 2, 0))  # 调整为 HWC 格式
        label = labels[i].item()  # 获取标签（标量）

        ax.imshow(image)  # 显示图像
        ax.set_title(f"Label: {label}")  # 标注标签
        ax.axis('off')  # 关闭坐标轴

    plt.tight_layout()
    plt.show()


def save_teacher_logits(config):
    train_loader, val_loader, test_loader = load_medmnist(config, shuffle=False)
    teacher = ViTTeacher(config.class_num).to(DEVICE)
    teacher.eval()
    # Tensors to store the images and teacher logits
    # labels_list = []
    labels_tensor = torch.empty((len(train_loader.dataset), config.class_num))
    # Loop through the dataset and collect the necessary data
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(train_loader, desc="Extracting logits from teacher")):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Get teacher logits
            teacher_logits = teacher(images)  # (batch_size, num_classes)

            # Append images, logits, and labels
            # labels_list.append(labels)
            labels_tensor[i * config.batchsize: min(labels_tensor.shape[0], (i + 1) * config.batchsize), :] = teacher_logits

    # Concatenate all batches into a single tensor for each
    # labels_tensor = torch.cat(labels_list, dim=0)  # Shape: (N,)

    # Create a dataset that combines images and precomputed logits
    print("FINISH")



def load_teacher_logits_tensor(dataset_name, teacher="resnet50"):
    # read in logits from ResNet50
    # dir = "/Users/fructuswhite/courses/24fall/Computer-Vision/final-pj/distillation/logits"
    # dir = os.path.dirname(os.path.realpath('__file__')) + "/logits"
    dir = os.getcwd() + "/logits"
    # print(dir)
    logits = None
    if teacher == "resnet50":
        for filename in os.listdir(dir):
            if filename.startswith(dataset_name) and filename.endswith(".csv"):
                print(f"Find logits: {filename}")
                p = os.path.join(dir, filename)
                df = pd.read_csv(p)
                df = df.to_numpy()
                logits = torch.tensor(df, dtype=torch.float32)
    elif teacher == "vit":
        for filename in os.listdir(dir):
            if filename.startswith(dataset_name) and filename.endswith("_vit.npz"):
                print(f"Find logits: {filename}")
                p = os.path.join(dir, filename)
                logits = torch.load(p)
    else:
        raise Exception(f"teacher {teacher} not found.")
    if logits is None:
        raise Exception(f"logits not found for teacher: {teacher} and dataset {dataset_name}.")
    return logits



