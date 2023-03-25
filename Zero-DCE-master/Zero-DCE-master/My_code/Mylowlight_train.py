import argparse

import torch
import torch.optim
import os

from tensorboardX import SummaryWriter

import Mylossr
import Mymodel
import Mydataloader
import argparse

"初始化权重"
# m.weight.data是卷积核参数, m.bias.data是偏置项参数


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    '设置可用显卡'
    writer = SummaryWriter('./logs')
    DCE_net = Mymodel.enhance_net_nopool().cuda()
    '定义网络'
    #init_img = torch.rand(1, 3, 256, 256).cuda()
    #writer.add_graph(DCE_net, (init_img,))


    DCE_net.apply(weights_init)
    '# 递归的调用weights_init函数,遍历DCE_net的submodule作为参数'
    'apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上'

    if config.load_pretrain ==True:

        DCE_net.load_state_dict(torch.load(config.pretrain_dir))

        #DCE_net.load_state_dict(torch.load(config.pretrain_dir))
    '加载预训练模型'
    train_dataset = Mydataloader.lowlight_loader(config.lowlight_images_path)

    '加载训练数据集'
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,num_workers =config.num_workers, pin_memory=True)
    '加载数据的数据集/每个batch加载多少样本/在每个epoch重新打乱数据/用多少个子进程加载数据/将内存的Tensor转义到GPU的显存就会更快一些'
    L_color = Mylossr.L_color()
    L_spa = Mylossr.L_spa()
    L_exp = Mylossr.L_exp(16,0.6)
    L_TV = Mylossr.L_TV()

    optimizer=torch.optim.Adam(DCE_net.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    '用于迭代优化的参数或者定义参数组/学习率/权重衰减'

    DCE_net.train()


    for epoch in range(config.num_epochs):
        for iteration,img_lowlight in enumerate(train_loader):
            '返回下标和对应的值'

            img_lowlight=img_lowlight.cuda()

            enhanced_image_1,enhanced_image,A=DCE_net(img_lowlight)
            Loss_TV=200*L_TV(A)
            Loss_spa=torch.mean(L_spa(enhanced_image,img_lowlight))
            Loss_col=5*torch.mean(L_color(enhanced_image))
            Loss_exp=10*torch.mean(L_exp(enhanced_image))

            loss=Loss_TV+Loss_col+Loss_spa+Loss_exp
            '计算损失'
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(),config.grad_clip_norm)
            '解决梯度爆炸问题'
            optimizer.step()
            '进行梯度更新'
            writer.add_scalar('train_loss_My_zero_dce_new3', loss, epoch + 1)
            if((iteration+1)%config.display_iter)==0:
                print("Loss at iteration",iteration+1,":",loss.item())
                '取出张量具体位置的元素元素值'

            if ((iteration + 1) % 250) == 0:
                print("全部训练集训练完", epoch + 1, "次")
            writer.close()

        if((epoch+1)%5==0):
            torch.save(DCE_net.state_dict(),config.snapshot_folder+ "317epoch"+str(epoch+1)+'.pth')


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    #输入参数
    parser.add_argument('--lowlight_images_path',type=str,default="E:\\zero-dce\\Zero-DCE-master\\Zero-DCE-master\\Zero-DCE_code\\data\\train_data\\")
    parser.add_argument('--pretrain_dir',type=str,default="snapshots/Epoch99.pth")
    parser.add_argument('--train_batch_size',type=int,default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--weight_decay',type=float,default=0.0001) #权重衰减
    parser.add_argument('--grad_clip_norm',type=float,default=0.1)  #解决梯度爆炸的参数
    parser.add_argument('--num_epochs',type=float,default=20)     #200次
    parser.add_argument('--val_batch_size',type=int,default=4)
    parser.add_argument('--display_iter',type=int,default=10)     #多少轮打印一次
    parser.add_argument('--snapshot_iter',type=int,default=10)
    parser.add_argument('--snapshot_folder',type=str,default="snapshots/")
    parser.add_argument('--load_pretrain',type=bool,default=False)  #是否加载预训练模型

    config=parser.parse_args()

    if not os.path.exists(config.snapshot_folder):
        os.mkdir(config.snapshot_folder)

    train(config)
