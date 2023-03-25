from tensorboardX import SummaryWriter
from PIL import  Image
import numpy as np


writer=SummaryWriter("logs")
'在给定目录中创建事件文件，并向其中添加摘要和事件'
image_path="E:\\zero-dce\\Zero-DCE-master\\Zero-DCE-master\\Mypytorch\\dataset\\bees\\2low.png"
img=Image.open(image_path)
img_array=np.array(img)
writer.add_image("train",img_array,2,dataformats='HWC')
print(img_array.shape)

# writer.all_image()
for i in range(100):
    writer.add_scalar("y=2x",3*i,i)

writer.close()
#打开方式 tensorboard --logdir=logs

