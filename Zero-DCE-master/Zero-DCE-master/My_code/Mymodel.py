import torch
import torch.nn as nn


class enhance_net_nopool(nn.Module):
    # 在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        '用nn.Module的初始化方法来初始化继承的属性'

        self.relu = nn.ReLU(inplace=True)
        '将会改变原有的值'

        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True)
        '128?'
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)

        self.e_conv7=nn.Conv2d(number_f*2,24,3,1,1,bias=True)

        self.maxpool=nn.MaxPool2d(2,stride=2,return_indices=False,ceil_mode=False)
        'return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助  ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整'
        self.upsample=nn.UpsamplingBilinear2d(scale_factor=2)
        'scale_factor=2 输出对于输入放大的倍数 '

    def forward(self, x):
        x1=self.relu(self.e_conv1(x))
        x2=self.relu(self.e_conv2(x1))
        x3=self.relu(self.e_conv3(x2))
        x4=self.relu(self.e_conv4(x3))

        x5=self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        '?在第一维度'
        x6=self.relu(self.e_conv6(torch.cat([x2,x5],1)))


        ren=self.e_conv7(torch.cat([x1,x6],1))
        renx=torch.cat([x1,x6],1)
        x_r=torch.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        r1,r2,r3,r4,r5,r6,r7,r8=torch.split(x_r,3,dim=1)
        "当split_size为一个int数时，若不能整除int，剩余的数据直接作为一块"

        "以下存疑"
        x=x+r1*(torch.pow(x,2)-x)
        x=x+r2*(torch.pow(x,2)-x)
        x=x+r3*(torch.pow(x,2)-x)
        enhance_image_1=x+r4*(torch.pow(x,2)-x)
        x=enhance_image_1+r5*(torch.pow(enhance_image_1,2)-enhance_image_1)
        x=x+r6*(torch.pow(x,2)-x)
        x=x+r7*(torch.pow(x,2)-x)
        enhance_image=x+r8*(torch.pow(x,2)-x)

        r=torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
        return enhance_image_1,enhance_image,r
