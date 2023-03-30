import torch
import torchvision

#加载方式1
vgg_model=torch.load("vgg16_method1.pth")



#加载方式2
vgg_model=torch.load("vgg16_method2.pth")

#打印的数值
print(vgg_model)

#打印的结构
vgg16=torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(vgg_model)
print(vgg16)

