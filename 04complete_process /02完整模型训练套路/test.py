"""
@Description:  测试 验证 demo https://www.bilibili.com/video/BV1hE411t7RN?p=32
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : test
@Time     : 2022/5/23 下午10:38
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/23 下午10:38        1.0             None
"""
import torch
import torchvision.transforms
from PIL import Image
from torch import nn

image_path = "../../imgs/dog.jpeg"
image = Image.open(image_path)
image = image.convert("RGB")
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape) # torch.Size([3, 32, 32])

# 2 model
class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(), #展躺
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10),

        )

    def forward(self, x):
        output = self.model(x)
        return output

model = torch.load("../../pth/module_all/module_6.pth")
print(model)
image = torch.reshape(image, (1,3,32,32)) # !!!!!!!reshape

model.eval()
with torch.no_grad():
    output = model(image)
print(output.argmax())
# 分类结果 {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}