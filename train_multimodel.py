import torch
import torch.optim as optim
import numpy as np
from multi_model import MutilModalClassifier

def train_multi_model(model, optimizer, criterion, train_loader, valid_loader, device):  
    # 训练轮次
    num_epochs = 20
    best_accuracy = 0
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
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
        
        # 如果验证准确率提高，则保存模型
        if valid_acc > best_accuracy:
            print('Saving model...')
            best_accuracy = valid_acc
            torch.save(model.state_dict(), './best_multimodal.pth')
            


def evaluate_model(model, criterion, data_loader, device):
    model.eval()  # 设置模型为评估模式
    total_loss, correct_preds = 0.0, 0
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
            correct_preds += torch.sum(predicted == targets.data)  # 累积准确预测数
    return total_loss / total_batches, correct_preds / len(data_loader.dataset)


def test_multi_model(test_loader, device):
    model = MutilModalClassifier()
    model.load_state_dict(torch.load('./best_multimodal.pth', map_location=device))
    model.to(device)
    model.eval()  
    guids = []
    preds = []
    with torch.no_grad():
        for batch_idx, (images, *texts, targets) in enumerate(test_loader):
            images = images.to(device)
            texts = [text.to(device) for text in texts]
            output = model(images, *texts)
            _, pred = torch.max(output, 1)
            guids.extend(targets.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())
    
    with open('./result.txt', 'w') as f:
        f.write('guid,tag\n')
        for guid, pred in zip(guids, preds):
            if pred == 0:
                pred_label = 'negative'
            elif pred == 1:
                pred_label = 'positive'
            elif pred == 2:
                pred_label = 'neutral'
            f.write(f'{guid},{pred_label}\n')


def run_multi_model(learning_rate, momentum, train_loader, valid_loader, test_loader):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Get device (GPU or CPU)
    model = MutilModalClassifier()  # Initialize model
    print('training...')  
    model.to(device)  
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  
    criterion = torch.nn.CrossEntropyLoss()  # Initialize loss function
    train_multi_model(model, optimizer, criterion, train_loader, valid_loader, device)  
    test_multi_model(test_loader, device)  
