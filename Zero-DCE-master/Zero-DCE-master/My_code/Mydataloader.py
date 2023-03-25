import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random


"产生随机种子"
random.seed(1143)

"填充训练列表"
def populate_train_list(lowlight_images_path):
    "返回所有匹配的文件路径列表（list）"
    image_list_lowlight=glob.glob(lowlight_images_path+"*.jpg")

    train_list=image_list_lowlight
    "用于将一个列表中的元素打乱 "
    random.shuffle(train_list)

    return train_list

"一个抽象类。所谓抽象类就是类的抽象化，而类本身就是不存在的，所以抽象类无法实例化。它存在的意义就是被继承。而且继承抽象类的类必须要重写抽象类的方法"
"用户想要加载自定义的数据只需要继承这个类，并且覆写其中的两个方法 len 和 get item"
class lowlight_loader(data.Dataset):
    def __init__(self, lowlight_images_path):

        self.train_list=populate_train_list(lowlight_images_path)
        self.size=256

        self.data_list=self.train_list
        print("total training example: ",len(self.train_list))


    def __getitem__(self, index):
        data_lowlight_path =self.data_list[index]

        data_lowlight=Image.open(data_lowlight_path)
        "返回的是PIL类型"

        data_lowlight = data_lowlight.resize((self.size,self.size),Image.ANTIALIAS)
        "Image.ANTIALIAS代表高质量"
        data_lowlight = (np.asarray(data_lowlight)/255.0)
        "转换为数组array"
        data_lowlight = torch.from_numpy(data_lowlight).float()
        '数组转为张量'

        return data_lowlight.permute(2,0,1)
    '交换tensor维度  tensor的高纬矩阵为N C H W  array的高维矩阵为H W C'

    def __len__(self):
        return len(self.data_list)
