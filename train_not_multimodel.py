import torch
from multi_model import MutilModalClassifier
import torch.optim as optim
import numpy as np

def train_model(model, optimizer, train_loader, valid_loader, device, model_type):
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 20  # 训练轮次
    model.train()  # 设置模型为训练模式
    for epoch in range(1, num_epochs + 1): 
        # 初始化训练和验证损失和准确率
        train_acc_sum, train_loss, valid_acc_sum, valid_loss = 0.0, 0.0, 0.0, 0.0
        train_batches = len(train_loader)
        valid_batches = len(valid_loader)
        for batch_pictures, *batch_texts, batch_labels in train_loader:
            # 将数据移到设备上
            batch_pictures = batch_pictures.to(device)
            batch_texts = [item.to(device) for item in batch_texts]
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()  # 清除梯度
            output = model(batch_pictures, *batch_texts)  # 前向传播
            loss = criterion(output, batch_labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            train_loss += loss.item()  # 累积训练损失
            _, preds = torch.max(output, 1)  # 获取预测结果
            train_acc_sum += torch.sum(preds == batch_labels.data)  # 计算训练准确率
        
        # 计算平均训练损失和准确率
        train_loss /= train_batches
        train_acc = train_acc_sum / len(train_loader.dataset)
        
        # 在验证集上评估模型
        valid_loss, valid_acc = evaluate_model(model, criterion, valid_loader, device)
            
        # 打印轮次统计信息
        print(f'Model Type: {model_type}\t Epoch: {epoch}\t Train Loss: {train_loss:.4f}\t Train Acc: {train_acc:.4f}\t Valid Loss: {valid_loss:.4f}\t Valid Acc: {valid_acc:.4f}')


def evaluate_model(model, criterion, data_loader, device):
    # 将模型设置为评估模式
    model.eval()  
    total_loss, correct_predictions = 0.0, 0
    total_batches = len(data_loader)
    for batch_idx, (images, *texts, targets) in enumerate(data_loader):
        # 将数据移到设备上
        images = images.to(device)
        texts = [text.to(device) for text in texts]
        targets = targets.to(device)
        with torch.no_grad():  # 禁用梯度追踪
            outputs = model(images, *texts)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total_loss += loss.item()  # 累积损失
            correct_predictions += torch.sum(predicted == targets.data)  # 累积准确预测数
    return total_loss / total_batches, correct_predictions / len(data_loader.dataset)


def run_model(learning_rate, momentum, train_loader, valid_loader, test_loader, model_type):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = MutilModalClassifier(model_type)
    
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    train_model(model, optimizer, train_loader, valid_loader, device, model_type)

