"""
@Description:  https://www.bilibili.com/video/BV1hE411t7RN?p=10&spm_id_from=pageDriver
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : transforms
@Time     : 2022/5/24 下午9:38
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/24 下午9:38        1.0             None
"""
from PIL import Image
from torchvision import transforms

img_path = "../imgs/dog.jpeg"
# PIL转tensor
img = Image.open(img_path)
print(img)
tensor_trans = transforms.ToTensor() # 转成Tensor
tensor_img = tensor_trans(img)
print(tensor_img.shape) # torch.Size([3, 187, 280])

# numpy转tensor
import cv2
img = cv2.imread(img_path)
print(type(img), img.shape) # (187, 280, 3)
tensor_trans = transforms.ToTensor() # 转成Tensor
tensor_img = tensor_trans(img)
print(tensor_img.shape) # torch.Size([3, 187, 280])

