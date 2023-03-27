import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10("./dataset2",train=False,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
#drop+last  是否整除舍弃数据  shuffle 是否打乱

#测试数据集中的第一张图片
img,target=test_data[0]
print(img.shape)
print(target)


writer=SummaryWriter("dataloader")

for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,targets=data
        print(imgs.shape)
        print(targets)
        writer.add_images("epoch:{}".format(epoch),imgs,step)
        step=step+1

writer.close()