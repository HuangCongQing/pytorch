# import torch
# print(torch.cuda.is_available())
import torch
from torch import nn

class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


mymodule = Mymodule() # 调用__init__
x = torch.tensor(1.0)
output = mymodule(x) # 调用forward
print(output)