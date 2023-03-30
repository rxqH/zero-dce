import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("./dataset2",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)


class seq(nn.Module):
    def __init__(self):
        super(seq, self).__init__()
        self.module1=nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self,x):
        x=self.module1(x)
        return x

loss=nn.CrossEntropyLoss()
rxq=seq()


for data in dataloader:
    imgs,target=data
    outputs=rxq(imgs)
    result_loss=loss(outputs,target)
    result_loss.backward()
    print("ok")