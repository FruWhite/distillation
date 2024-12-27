import datetime

from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
import wandb
import numpy as np
from dataload import load_teacher_logits_tensor


def kd_epoch(student_model, teacher_model, train_loader, optimizer, loss_fn, config, logits=None):
    DEVICE = config.device
    student_model.train()
    total_loss = 0
    if not config.use_saved_teacher_logits:
        if config.teacher == "resnet50":
            raise Exception(f"teacher {config.teacher} is not available in local now. please use saved logits.\n(set config.use_saved_teacher_logits to True)")

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher_model(images)

            student_logits = student_model(images)

            loss = loss_fn(student_logits, teacher_logits, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    else:
        logits.to(DEVICE)
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            teacher_logits = logits[i * config.batchsize: min(logits.shape[0], (i + 1) * config.batchsize), 1:].to(DEVICE)

            # 学生模型输出
            student_logits = student_model(images)

            # print(f"shape t logit: {teacher_logits.shape}, s logit : {student_logits.shape}")

            # ad-hoc式修复最后不够1batch时奇怪的shape mismatch问题

            if teacher_logits.shape != student_logits.shape:
                m = min(teacher_logits.shape[0], student_logits.shape[0])
                teacher_logits = teacher_logits[:m, :]
                student_logits = student_logits[:m, :]
                labels = labels[:m, :]
                # print(teacher_logits.shape)
                # print(student_logits.shape)
                # print(labels.shape)

            if config.dataset_name == "chestmnist" or labels.shape[1] > 1:  # one-hot label
                labels = torch.argmax(labels, -1)


            # 计算损失
            loss = loss_fn(student_logits, teacher_logits, labels)
            total_loss += loss.item()

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_loss / len(train_loader)


def evaluate(model, test_loader, config):
    DEVICE = config.device
    model.eval()
    correct = 0
    total = 0

    # 添加进度条
    loop = tqdm(test_loader, desc="Evaluating", ncols=100)
    with torch.no_grad():
        for images, labels in loop:
            labels = labels.squeeze()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(accuracy=correct / total)
    return correct / total


def kd_train(student_model, teacher_model, train_loader, val_loader, optim, distillation_loss, config):
    EPOCHS = config.epochs

    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=config.project_name, config=config.__dict__, name=nowtime, save_code=True)
    student_model.run_id = wandb.run.id
    student_model.best_metric = -1.
    logits = None 
    if config.use_saved_teacher_logits:
        logits = load_teacher_logits_tensor(config.dataset_name, teacher=config.teacher)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = kd_epoch(student_model, teacher_model, train_loader, optim, distillation_loss, config, logits)
        val_accuracy = evaluate(student_model, val_loader, config)
        print(f"Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        if val_accuracy > student_model.best_metric:
            student_model.best_metric = val_accuracy
        wandb.log({'epoch': epoch + 1, "val_acc": val_accuracy, "best_val_acc": student_model.best_metric})
    wandb.finish()
    return student_model


def train_epoch(model, train_loader, optimizer, criterion, config):
    DEVICE = config.device
    model.train()  # Set the model to training mode
    running_loss = 0
    correct_preds = 0
    total_preds = 0
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=config.project_name, config=config.__dict__, name=nowtime, save_code=True)
    loop = tqdm(train_loader, desc="Training", ncols=100)
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward pass
        outputs = model(images)
        labels = labels.squeeze()

        try:
            loss = criterion(outputs, labels)
        except ValueError:
            print(f"{labels.size()}, {outputs.size()}")
            optimizer.zero_grad()
            continue
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_preds / total_preds
    return epoch_loss


def base_train(student_model, train_loader, val_loader, optim, config):
    EPOCHS = config.epochs

    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=config.project_name, config=config.__dict__, name=nowtime, save_code=True)
    student_model.run_id = wandb.run.id
    student_model.best_metric = -1.
    logits = None

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(student_model, train_loader, optim, nn.CrossEntropyLoss(), config)
        val_accuracy = evaluate(student_model, val_loader, config)
        print(f"Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        if val_accuracy > student_model.best_metric:
            student_model.best_metric = val_accuracy
        wandb.log({'epoch': epoch + 1, "val_acc": val_accuracy, "best_val_acc": student_model.best_metric})
    wandb.finish()
    return student_model





