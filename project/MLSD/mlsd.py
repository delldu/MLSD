import os
import torch
import torch.nn as nn
from  torch.nn import  functional as F
from typing import Tuple, List

import pdb

class BlockTypeA(nn.Module):
    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale = True):
        super(BlockTypeA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c2, out_c2, kernel_size=1),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, kernel_size=1),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True)
        )
        self.upscale = upscale

    def forward(self, a, b):
        b = self.conv1(b)
        a = self.conv2(a)
        if self.upscale:
             b = F.interpolate(b, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.cat((a, b), dim=1)


class BlockTypeB(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x

class BlockTypeC(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        self.channel_pad = out_planes - in_planes
        self.stride = stride
        #padding = (kernel_size - 1) // 2

        # TFLite uses slightly different padding than PyTorch
        if stride == 2:
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)


    def forward(self, x):
        if self.stride == 2:
            x = F.pad(x, (0, 1, 0, 1), "constant", 0.0)

        for module in self:
            if not isinstance(module, nn.MaxPool2d):
                x = module(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=False):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        width_mult = 1.0
        round_nearest = 8

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            #[6, 160, 3, 2],
            #[6, 320, 1, 1],
        ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(4, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features = nn.Sequential(*features)
        self.fpn_selected = [1, 3, 6, 10, 13]
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        # torch.jit.script(self)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fpn_features = []
        for i, f in enumerate(self.features): #  len(self.features) -- 14
            x = f(x)
            if i in self.fpn_selected:
                fpn_features.append(x)

        c1, c2, c3, c4, c5 = fpn_features
        # (Pdb) c1.size() -- [1, 16, 256, 256]
        # (Pdb) c2.size() -- [1, 24, 128, 128]
        # (Pdb) c3.size() -- [1, 32, 64, 64]
        # (Pdb) c4.size() -- [1, 64, 32, 32]
        # (Pdb) c5.size() -- [1, 96, 32, 32]

        return c1, c2, c3, c4, c5



class MobileV2_MLSD_Large(nn.Module):
    def __init__(self):
        super(MobileV2_MLSD_Large, self).__init__()

        self.backbone = MobileNetV2()
        ## A, B
        self.block15 = BlockTypeA(in_c1= 64, in_c2= 96, out_c1= 64, out_c2=64, upscale=False)
        self.block16 = BlockTypeB(128, 64)

        ## A, B
        self.block17 = BlockTypeA(in_c1 = 32,  in_c2 = 64, out_c1= 64,  out_c2= 64)
        self.block18 = BlockTypeB(128, 64)

        ## A, B
        self.block19 = BlockTypeA(in_c1=24, in_c2=64, out_c1=64, out_c2=64)
        self.block20 = BlockTypeB(128, 64)

        ## A, B, C
        self.block21 = BlockTypeA(in_c1=16, in_c2=64, out_c1=64, out_c2=64)
        self.block22 = BlockTypeB(128, 64)

        self.block23 = BlockTypeC(64, 16)

        self.score_threshold = 0.10 # Hough value threshold
        self.distance_threshold = 20.0 # Hough distance threshold

        self.load_weights()


    def forward(self, x):
        B, C, H, W = x.size()
        assert B == 1 and C == 4, "Now only support one batch and rgba channels"
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=True)
        x = (x - 0.5) * 2.0 # convert x from [0.0, 1.0] to [-1.0, 1.0]

        c1, c2, c3, c4, c5 = self.backbone(x)

        x = self.block15(c4, c5)
        x = self.block16(x)

        x = self.block17(c3, x)
        x = self.block18(x)

        x = self.block19(c2, x)
        x = self.block20(x)

        x = self.block21(c1, x)
        x = self.block22(x)
        x = self.block23(x) # x.size() -- [1, 16, 256, 256]
        x = x[:, 7:, :, :]  # x.size() -- [1,  9, 256, 256]

        center = x[:, 0, :, :] # size() -- [1, 256, 256]
        displacement = x[:, 1:5, :, :][0].permute(1, 2, 0) # size() -- [4, 256, 256] ==> [256, 256, 4]

        ksize = 3
        heat = torch.sigmoid(center) # size() -- [1, 256, 256]
        hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize-1)//2) # size() -- [1, 256, 256]
        heat = heat * (hmax == heat).float() # keep local maximum
        heat = heat.reshape(-1, ) # size() -- [65535]
        scores, coords = torch.topk(heat, 200, dim=0, largest=True) # size() -- [topk -- 200]

        yy = torch.floor_divide(coords, 256)
        xx = torch.fmod(coords, 256)

        x1y1 = displacement[:, :, :2]
        x2y2 = displacement[:, :, 2:]
        distance_map = torch.sqrt(torch.sum((x1y1 - x2y2) ** 2, dim=2)) # size() -- [256, 256]

        h_ratio = H/256.0 # orignal heat size -- [1, 256, 256]
        w_ratio = W/256.0

        # Setup [0.0, 0.0, 0.0, 0.0] for make sure torch.stack(segment_list, dim = 0) OK !!!
        segment_list: List[torch.Tensor] = [torch.tensor([0.0, 0.0, 0.0, 0.0]).to(x.device)]
        for y, x, s in zip(yy, xx, scores):
            d = distance_map[y, x]
            if s > self.score_threshold and d > self.distance_threshold:
                x1 = (x + displacement[y, x, 0]) * w_ratio
                y1 = (y + displacement[y, x, 1]) * h_ratio
                x2 = (x + displacement[y, x, 2]) * w_ratio
                y2 = (y + displacement[y, x, 3]) * h_ratio
                segment_list.append(torch.stack((x1, y1, x2, y2), dim=0))

        lines = torch.stack(segment_list, dim = 0)

        return lines


    def load_weights(self, model_path="models/MLSD.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading weight from {checkpoint} ...")
            self.load_state_dict(torch.load(checkpoint))
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"Weight file '{checkpoint}' not exist !!!")
