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
                #dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # 做 1 x 1 的逐层卷机，进行特征融合
                #pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            #需要升纬的时候
            self.conv = nn.sequential(
                # 做 1 x 1 的逐层卷机，进行特征降维，通道数上升，更多的特征表征能力
                #pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # 做 3 x 3 的逐层卷机，进行跨特征点的特征提取，通道数不变，跨特征点的特征读取
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # 做 1 x 1 的逐层卷机，进行特征融合。通道数下降，方便运算
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

#-------------------------------------#
# define the MobileNetV2, which is the backbone of deeplabv3+. 
# It use inverted residual block to replace the standard residual block as the basic unit.
#-------------------------------------#
class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size = 224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            #expand_radio, output_channel, repeat_time, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        assert input_size%32 == 0 
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel* width_mult) if width_mult>1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url,model_dir=model_dir)

def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(n_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'), strict=False)
    return model

if __name__ == "__main__":
    model = mobilenetv2()
    for i, layer in enumerate(model.features):
        print(i, layer)

        