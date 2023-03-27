import torch
from torch import nn


#重写module中的init和forward
class firstmodule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output=x+1
        return output

rxq=firstmodule()
x=torch.tensor(1.0)
output=rxq(x)
print(output)