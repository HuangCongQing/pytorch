'''
Description: 参考 https://blog.csdn.net/weixin_43656490/article/details/128396939
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2023-03-23 15:23:08
LastEditTime: 2023-03-23 15:31:08
FilePath: /pytorch/onnx/01_VFE_op_onnx.py
'''
import torch
import torch.nn as nn
from torch.autograd import Function

class ScatterMax(Function):
    @staticmethod
    def forward(ctx, src):
        temp = torch.unique(src)
        # print(src.shape)
        # print(temp.shape)
        out = torch.zeros((temp.shape[0], src.shape[1]), dtype=torch.float32, device=src.device)
        return out
    @staticmethod
    def symbolic(g, src):
        return g.op("ScatterMaxPlugin", src) # op名字

class VFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn_layer0 = nn.Sequential(
            nn.Linear(in_features=10, out_features=64, bias=False),
            nn.BatchNorm1d(num_features=32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        )
        # self.scatter = ScatterMax()
    def forward(self, x):
        x = self.pfn_layer0(x)
        x = ScatterMax.apply(x) # 调用apply<<<<<<<<<<<
        return x

if __name__ == '__main__':
    pillarvfe = VFE()
    input = torch.zeros((40000, 32, 10))
    output = pillarvfe(input)
    # print(output.shape)

    torch.onnx.export(pillarvfe,
                      input,
                      "vfe.onnx",
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=True,
                      keep_initializers_as_inputs=True,
                      input_names=["input"],
                      output_names=["output"],
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
