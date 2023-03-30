import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader


dataset=torchvision.datasets.CIFAR10("./dataset2",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)



# input = torch.tensor([[1,2,0,3,1],
#                     [0,1,2,3,1],
#                     [3,1,0,4,3],
#                     [0,6,2,1,3],
#                     [0,1,2,3,1]],dtype=torch.float32)
#
# input=torch.reshape(input,(-1,1,5,5))
# class Rxq(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.maxpool=MaxPool2d(kernel_size=3,ceil_mode=True)
#     def forward(self, x):
#         output = self.maxpool(x)
#         return output
#
# rxq=Rxq()
# writer=SummaryWriter("./nn.maxpool")
#
# step=0
# for data in dataloader:
#     imgs,target=data
#     output=rxq(imgs)
#     writer.add_images("output",output,step)
#     step= step+1
# writer.close()



input1=torch.tensor([[-0.5,5],
                     [2,-5]])
relu=nn.ReLU()
output1=relu(input1)
print(output1)
