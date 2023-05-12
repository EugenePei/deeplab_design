import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BatchNorm2d = nn.BatchNorm2d

# define the conv blocks
def conv_bn(inp, oup, stride):
    return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
            )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp,oup, 1, 1, 0,bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace = True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        # 如果stride 不为【1，2】则终止程序
        assert stride in [1,2]   

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.sequential(
                # 做 3 x 3 的逐层卷机，进行跨特征点的特征提取
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # 做 1 x 1 的逐层卷机，进行特征融合
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )