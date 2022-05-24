"""
@Description: tensorboard https://www.bilibili.com/video/BV1hE411t7RN?p=8&t=329.3
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : test_tb
@Time     : 2022/5/24 下午4:54
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/24 下午4:54        1.0             None
"""
from torch.utils.tensorboard import SummaryWriter

# 按住ctrl. 鼠标点击查看源码
writer = SummaryWriter("../logs/tb_test")
# writer.add_scalar() # 标量（数）
# writer.add_image() # 图像

# 1 saclar
for i in range(100):
    writer.add_scalar("y=x", i, i)

# 2 img
img_path = "../imgs/dog.jpeg"
import cv2
img = cv2.imread(img_path)
print(type(img)) # numpy格式
writer.add_image("img", img, 1, dataformats='HWC')

writer.close()
print("结束")
