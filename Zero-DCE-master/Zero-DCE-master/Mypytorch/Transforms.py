from tensorboardX import SummaryWriter
from torchvision import transforms
from PIL import Image

image_path="E:\\zero-dce\\Zero-DCE-master\\Zero-DCE-master\\Mypytorch\\dataset\\ants\\1.jpg"
image_PIL=Image.open(image_path)
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(image_PIL)

writer=SummaryWriter("logs")

writer.add_image("tensor_img",tensor_img)
writer.close()