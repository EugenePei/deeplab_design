import torch
import torch.nn as nn
import torch.nn.functional as F 



class MobileNetV2(nn.Mobel):
    def __init__(self, downsample_factor=8, pretained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        