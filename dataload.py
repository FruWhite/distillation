import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def load_medmnist(config, dataset_name='bloodmnist'):
    # 获取数据集的相关信息
    BATCH_SIZE = config.batchsize
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])

    # 数据增强及预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])  # 对数据进行归一化
    ])

    # 加载数据集
    train_dataset = DataClass(split='train', transform=transform, download=True)
    val_dataset = DataClass(split='val', transform=transform, download=True)
    test_dataset = DataClass(split='test', transform=transform, download=True)

    # 构建 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
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




