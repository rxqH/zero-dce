import torch
import torchvision
from PIL import Image
from cv2 import transform
from torch import nn

image_path = 'dataset/dog.png'

image = Image.open(image_path)


transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                          torchvision.transforms.ToTensor()
                                          ])

img = transform(image)

img = torch.reshape(img,(1,3,32,32))
img = img.cuda()

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


model = torch.load("rxq_1.pth")

model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))