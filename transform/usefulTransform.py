"""
@Description: 常用的transform https://www.bilibili.com/video/BV1hE411t7RN?p=13&t=431.6
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : usefulTransform
@Time     : 2022/5/24 下午10:11
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/24 下午10:11        1.0             None
"""
from PIL import Image
from torchvision import transforms

# 1 toTensor
img = Image.open("../imgs/dog.jpeg")
trans_totensor = transforms.ToTensor() # 实例化
img_tensor = trans_totensor(img)
print(img_tensor.shape)

# 2 Normalize
print(img_tensor[0][0][0])
trans_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 实例化
img_norm = trans_normalize(img_tensor)
print(img_norm[0][0][0])

# 3 Resize
print(img_tensor.shape) # torch.Size([3, 187, 280])
trans_resize = transforms.Resize((32, 32))
img_resize = trans_resize(img_tensor)
print(img_resize.shape) # torch.Size([3, 32, 32])

# 4 Compose（需要多步骤建议使用）
trans_compose = transforms.Compose([trans_resize])
img_compose = trans_compose(img_tensor)
print(img_compose.shape)

# RandomCrop
trans_randomcrop = transforms.RandomCrop((100, 100))
trans_crop = trans_randomcrop(img_tensor)
print(trans_crop.shape)
