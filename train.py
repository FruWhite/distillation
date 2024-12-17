import datetime

from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
DEVICE = "cuda"
import wandb

def kd_epoch(student_model, teacher_model, train_loader, optimizer, loss_fn, config, logits=None):
    student_model.train()
    teacher_model.eval()
    total_loss = 0
    loop = tqdm(train_loader, desc="Training", ncols=100)
    if not config.teacher_logits_available:
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 教师模型输出（无需梯度）
            with torch.no_grad():
                teacher_logits = teacher_model(images)

            # 学生模型输出
            student_logits = student_model(images)

            # 计算损失
            loss = loss_fn(student_logits, teacher_logits, labels)
            total_loss += loss.item()

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
    else:
        for i, (images, labels) in enumerate(loop):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            teacher_logits = logits[i * config.batchsize: min(logits.shape[0], (i + 1) * config.batchsize), :]

            # 学生模型输出
            student_logits = student_model(images)

            # 计算损失
            loss = loss_fn(student_logits, teacher_logits, labels)
            total_loss += loss.item()

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())


    return total_loss / len(train_loader)


def evaluate(model, test_loader):
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
    if config.teacher_logits_available:
        logits = torch.load(f"logits/{config.dataset_name}_vit.npz")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = kd_epoch(student_model, teacher_model, train_loader, optim, distillation_loss, config)
        val_accuracy = evaluate(student_model, val_loader)
        print(f"Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        if val_accuracy > student_model.best_metric:
            student_model.best_metric = val_accuracy
        wandb.log({'epoch': epoch + 1, "val_acc": val_accuracy, "best_val_acc": student_model.best_metric})
    wandb.finish()
    return student_model


def train_epoch(model, train_loader, optimizer, criterion, config):
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
        loss = criterion(outputs, labels)

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







