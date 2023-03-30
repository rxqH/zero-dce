
import torchvision
from torch import nn

vgg16=torchvision.models.vgg16(pretrained=False)
vgg16_two=torchvision.models.vgg16(pretrained=False)
#默认参数，未曾初始化
dataset = torchvision.datasets.CIFAR10("./dataset2",train=False,transform=torchvision.transforms.ToTensor())


#修改vgg16(添加)
vgg16.add_module('add_linear',nn.Linear(1000,10))

vgg16.classifier.add_module('add_linear',nn.Linear(1000,10))


#修改
vgg16_two.classifier[6]=nn.Linear(4096,10)
print(vgg16)
