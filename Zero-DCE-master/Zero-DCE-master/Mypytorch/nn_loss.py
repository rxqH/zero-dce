import torch
from torch import nn

input=torch.tensor([1,2,3],dtype=torch.float32)
output=torch.tensor([4,5,6],dtype=torch.float32)

inputs=torch.reshape(input,(1,1,1,3))
outputs=torch.reshape(output,(1,1,1,3))


loss_mse=nn.MSELoss()


loss=nn.L1Loss()
result=loss(input,output)
print(result)
