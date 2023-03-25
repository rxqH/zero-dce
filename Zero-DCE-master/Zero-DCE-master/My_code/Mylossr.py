import torch
import torch.nn as nn
import torch.nn.functional as  F
from torchvision.models.vgg import vgg16

"搜索设置自定义损失函数"
"采用继承nn.module的方法时候"
"定义一个损失函数类，继承自nn.Module，在forward中实现loss定义所有的数学操作使用tensor提供的math operation,返回的tensor是0-dim的scalar"

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()
    def forward(self, x):
        mean_rgb=torch.mean(x,[2,3],keepdim=True)
        mr,mg,mb=torch.split(mean_rgb,1,dim=1)
        Drg=torch.pow(mr-mg,2)
        Drb=torch.pow(mr-mb,2)
        Dbg=torch.pow(mb-mg,2)

        k=torch.pow(torch.pow(Dbg,2)+torch.pow(Drb,2)+torch.pow(Drg,2),0.5)
        '0.5?'
        return k


class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        kernel_left=torch.FloatTensor([[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right=torch.FloatTensor([[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up=torch.FloatTensor([[0,-1,0],[0,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down=torch.FloatTensor([[0,0,0],[0,1,0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        '自定义卷积核'

        self.weight_left=nn.Parameter(data=kernel_left,requires_grad=False)
        '以看作是一个类型转换函数，将一个不可训练的类型 Tensor 转换成可以训练的类型 parameter ?'
        self.weight_right=nn.Parameter(data=kernel_right,requires_grad=False)
        self.weight_up=nn.Parameter(data=kernel_up,requires_grad=False)
        self.weight_down=nn.Parameter(data=kernel_down,requires_grad=False)
        self.pool=nn.AvgPool2d(4)
    def forward(self, org, enhance):

        org_mean=torch.mean(org,1,keepdim=True)
        '???'
        enhance_mean=torch.mean(enhance,1,keepdim=True)

        org_pool=self.pool(org_mean)
        enhance_pool=self.pool(enhance_mean)

        D_org_left=F.conv2d(org_pool,self.weight_left,padding=1)
        D_org_right=F.conv2d(org_pool,self.weight_right,padding=1)
        D_org_up=F.conv2d(org_pool,self.weight_up,padding=1)
        D_org_down=F.conv2d(org_pool,self.weight_down,padding=1)

        D_enhance_left=F.conv2d(enhance_pool,self.weight_left,padding=1)
        D_enhance_right=F.conv2d(enhance_pool,self.weight_right,padding=1)
        D_enhance_up=F.conv2d(enhance_pool,self.weight_up,padding=1)
        D_enhance_down=F.conv2d(enhance_pool,self.weight_down,padding=1)

        D_left=torch.pow(D_org_left-D_enhance_left,2)
        D_right=torch.pow(D_org_right-D_enhance_right,2)
        D_up=torch.pow(D_org_up-D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E=(D_left+D_right+D_up+D_down)

        return E
        "存疑"

class L_exp(nn.Module):
    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        'patchsize 是大像素块的大小' \
        'mean_val 是默认经验值,描述亮度中间值'
        self.pool=nn.AvgPool2d(patch_size)
        self.mean_val=mean_val
    def forward(self, x):

        x=torch.mean(x,1,keepdim=True)
        mean=self.pool(x)

        d=torch.mean(torch.pow(mean-torch.FloatTensor([self.mean_val]).cuda(),2))
        '?'
        return d

#水平、垂直方向梯度平均值应很小
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight
    def forward(self, x):
        batch_size=x.size()[0]
        h_x=x.size()[2]
        w_x=x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)

        h_tv=torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv=torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h + w_tv/count_w)/batch_size

# class Sa_Loss(nn.Module):
#     def __init__(self):
#         super(Sa_Loss, self).__init__()
#
#     def