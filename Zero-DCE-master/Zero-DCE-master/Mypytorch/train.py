import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

from model import *


# 定义训练的设备
device = torch.device("cpu")
# device = torch.device("cuda")



# 准备数据集
train_data = torchvision.datasets.CIFAR10("./dataset2", train=True, transform=torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10("./dataset2", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#  创建网络模型
rxq = Rxq()
rxq = rxq.cuda()
#  方式二   rxq = rxq.to(device)


# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
# 优化器。
learning_rate = 0.01
optimizer = torch.optim.SGD(rxq.parameters(), lr=learning_rate)

# 设置训练网络的一些参数

# 记录训练的次数
total_train_step = 0
# 记录测试的次数。
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
# 新版的需要将=改成“”，这点巨坑……
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("----------第{}轮开始训练----------".format(i+1))
    rxq.train()  # 对某些特定的层起作用
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = rxq(imgs)
        loss = loss_fn(outputs, targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step+1
        if total_train_step % 100 ==0:
            print("训练次数：{}，loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤开始
    rxq.eval()   # 同样对某些特定层起作用
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = rxq(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss+loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy+accuracy.item()

    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",loss.item(),total_test_step)
    total_test_step = total_test_loss+1

    torch.save(rxq,"rxq_{}.pth".format(i+1))
    print("模型已保存")
writer.close()

