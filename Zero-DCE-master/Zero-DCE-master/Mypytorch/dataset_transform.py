import torchvision
from tensorboardX import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set=torchvision.datasets.CIFAR10(root="dataset2",train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root="dataset2",train=False,transform=dataset_transform,download=True)

# # print(test_set[0])
# img,target=test_set[0]  #图片，编号  编号代表类别
# print(img)

# print(test_set[0])

writer = SummaryWriter("dataset")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()


