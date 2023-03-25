from  torch.utils.data import  Dataset
import  cv2
from PIL import Image
import  os


class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.laber_dir=label_dir
        self.path_dir=os.path.join(self.root_dir,self.laber_dir)

        self.img_path=os.listdir(self.path_dir)
    def __getitem__(self, idx):
        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.root_dir,self.laber_dir,img_name)
        img=Image.open(img_item_path)
        return img

    def __len__(self):
        return len(self.img_path)

root_dir="E:\\zero-dce\\Zero-DCE-master\\Zero-DCE-master\\Mypytorch\\dataset"
laber_dir="ants"
laber_dir2="bees"
dataset=MyData(root_dir,laber_dir)
dataset2=MyData(root_dir,laber_dir2)
train_dataset=dataset+dataset2





