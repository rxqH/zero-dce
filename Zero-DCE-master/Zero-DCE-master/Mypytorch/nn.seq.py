import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


class Rxq(nn.Module):
    def __init__(self):
        super(Rxq, self).__init__()
        self.conv1=Conv2d(3,32,5,padding=2)
        self.maxpool=MaxPool2d(2)
        self.conv2=Conv2d(32,32,5,padding=2)
        self.maxpool2=MaxPool2d(2)
        self.conv3=Conv2d(32,64,5,padding=2)
        self.maxpool3=MaxPool2d(2)
        #64*4*4 =1024
        self.flatten=nn.Flatten()

        self.linear1=nn.Linear(1024,64)
        self.linear2=nn.Linear(64,10)
    def forward(self, x):
        x=self.conv1(x)
        x=self.maxpool(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.conv3(x)
        x=self.maxpool3(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        return x

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

# rxq=Rxq( )
# print(rxq)

rxq1=seq()
print(rxq1)

input=torch.ones(64,3,32,32)
output=rxq1(input)
print(output.shape)
# output=rxq(input)
# print(output.shape)

