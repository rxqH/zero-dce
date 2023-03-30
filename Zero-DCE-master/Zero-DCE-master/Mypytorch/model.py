#搭建神经网络
import torch
from torch import nn

#填充 步长也要写上
class Rxq(nn.Module):
    def __init__(self):
        super(Rxq, self).__init__()

        self.module=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self, x):
        x=self.module(x)
        return x


if __name__ == '__main__':
    input=torch.ones(64,3,32,32)
    rxq=Rxq()
    output=rxq(input)
    print(output.shape)