'''
Description: https://www.yuque.com/huangzhongqing/pytorch/ao4u9vp4xxs78t56
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2023-03-23 14:55:00
LastEditTime: 2023-03-23 15:18:07
FilePath: /pytorch/onnx/00_op_onnx.py
'''
import torch
import torch.nn as nn
from torch.autograd import Function
import onnx
import torch.onnx


class Requant_(Function):

    @staticmethod
    def forward(ctx, input, requant_scale, shift):  # ctx 必须要
        input = input.double(
        ) * requant_scale / 2**shift  # 为了等价于c中的移位操作。会存在int32溢出
        input = torch.floor(input).float()

        return torch.floor(input)

    @staticmethod
    def symbolic(g, *inputs):
        # 属性 scale_f=23.0, shift_i=8
        return g.op("Requant", inputs[0], scale_f=23.0, shift_i=8)


requant_ = Requant_.apply


class TinyNet(nn.Module):

    def __init__(self):
        super(TinyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(-1)
        x = requant_(x, 5, 5) # 自定义op
        return x


net = TinyNet().cuda()
ipt = torch.ones(2, 3, 12, 12).cuda()
torch.onnx.export(
    model=net, # model
    args=(ipt, ), # data
    f='tinynet.onnx', # 文件名
    verbose=False,
    opset_version=13,
    custom_opsets={"ai.onnx.contrib": 13},
    do_constant_folding=True,
    keep_initializers_as_inputs=True,
    input_names=["camera_feat"],
    output_names=['classifier_Feat0'],
    enable_onnx_checker=False # torch.onnx.utils.ONNXCheckerError: No Op registered for Requant with domain_version of 13
    )
print(onnx.load('tinynet.onnx'))
