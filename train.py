import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
train_losses = []
train_accuracies = []
test_accuracies = []


from pathlib import Path

from model import CNNModel

# 数据增强与预处理
transform = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),    # 随机仿射变换
    transforms.ToTensor(),  # numpy数组转为PyTorch Tensor, 像素值[0,255]->[0,1]
    transforms.Normalize((0.5,), (0.5,))    # [0,1]->[-1.1]
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)      # 每次取64张照片为一个批次，shuffle = True,训练时打乱顺序
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型并加载权重
model = CNNModel()

mnist_model = Path("mnist_model.pth")
if mnist_model.is_file():
    model.load_state_dict(torch.load("mnist_model.pth"))    # 可以接着上次继续训练

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()   # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)    # 自适应矩估计优化器

# 训练循环
epochs = 50
    # 遍历训练集
for epoch in range(epochs):
    running_loss = 0
    correct = 0
    total = 0
    # 遍历batch
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)    # 累加每个batch的损失
        _, predicted = torch.max(outputs, 1)    # 返回预测的数字
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 测试集评估
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)

    # 打印日志
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

# 保存模型
torch.save(model.state_dict(), "mnist_model.pth")

# 绘制曲线，左侧训练损失随epoch下降曲线，右侧训练准确率和测试准确率随epoch上升曲线
plt.figure(figsize=(10,5))

# Loss
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(test_accuracies, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy')
plt.legend()

plt.show()
