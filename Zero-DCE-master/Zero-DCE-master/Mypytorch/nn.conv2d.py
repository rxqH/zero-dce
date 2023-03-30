import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset2",train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64)

class rxqmoudle(nn.Module):
    def __init__(self):
        super(rxqmoudle, self).__init__()
        self.conv1 = Conv2d(3,6,3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return x

rxq=rxqmoudle()
print(rxq)

writer=SummaryWriter("./nn.conv2d")

step=0
for data in dataloader:
    imgs,targer=data
    output=rxq(imgs)
    #print(output.shape)
    writer.add_images("input",imgs,step)

    output=torch.reshape(output,(-1,3,30,30))
    #print(output.shape)
    writer.add_images("output",output,step)

    step=step+1

writer.close()