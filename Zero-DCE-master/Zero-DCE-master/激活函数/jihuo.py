import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def showsigmoid():
    plt.axis([-10, 10, 0, 1])
    ax = plt.gca()  # gca:get current axis得到当前轴
    # 设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # 挪动x，y轴的位置，也就是图片下边框和左边框的位置
    ax.spines['bottom'].set_position(('data', 0))  # data表示通过值来设置x轴的位置，将x轴绑定在y=0的位置
    ax.spines['left'].set_position(('axes', 0.5))
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)

    plt.show()

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def showtanh():
    plt.axis([-10, 10, -1, 1])
    ax = plt.gca()  # gca:get current axis得到当前轴
    # 设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # 挪动x，y轴的位置，也就是图片下边框和左边框的位置
    ax.spines['bottom'].set_position(('data', 0))  # data表示通过值来设置x轴的位置，将x轴绑定在y=0的位置
    ax.spines['left'].set_position(('axes', 0.5))
    x=x = np.arange(-10, 10, 0.1)
    y=tanh(x)
    plt.plot(x, y)

    plt.show()

def ReLU(x):

    return np.maximum(0, x)

def showReLU():
    plt.axis([-10, 10, -10, 10])
    ax = plt.gca()  # gca:get current axis得到当前轴
    # 设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # 挪动x，y轴的位置，也就是图片下边框和左边框的位置
    ax.spines['bottom'].set_position(('data', 0))  # data表示通过值来设置x轴的位置，将x轴绑定在y=0的位置
    ax.spines['left'].set_position(('axes', 0.5))
    x =np.arange(-10, 10, 0.1)
    y = ReLU(x)
    plt.plot(x, y)

    plt.show()

def LeakyRelu(x):
    return np.maximum(0.1*x,x)

def showLeakyRelu( ):
    plt.axis([-10, 10, -10, 10])
    ax = plt.gca()  # gca:get current axis得到当前轴
    # 设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # 挪动x，y轴的位置，也就是图片下边框和左边框的位置
    ax.spines['bottom'].set_position(('data', 0))  # data表示通过值来设置x轴的位置，将x轴绑定在y=0的位置
    ax.spines['left'].set_position(('axes', 0.5))
    x =np.arange(-10, 10, 0.1)
    y = LeakyRelu(x)
    plt.plot(x, y)

    plt.show()

showLeakyRelu()