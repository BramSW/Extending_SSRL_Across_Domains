import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
from lib.normalize import Normalize

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1_fonc(in_planes, out_planes=None, stride=1, bias=False):
    if out_planes is None:
        return nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class conv1x1(nn.Module):
    
    def __init__(self, planes, out_planes=None, stride=1):
        super(conv1x1, self).__init__()
        self.conv = conv1x1_fonc(planes)
    def forward(self, x):
        y = self.conv(x)
        return y

class conv_task(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, second=0):
        super(conv_task, self).__init__()
        self.second = second
        self.conv = conv3x3(in_planes, planes, stride)
        self.bns = nn.BatchNorm2d(planes)
        # print(type(self.bns))
    
    def forward(self, x):
        y = self.conv(x)
        # print(type(y), type(self.bns))
        if type(self.bns) == torch.nn.modules.batchnorm.BatchNorm2d:
            y = self.bns(y)
        else:
            y = self.bns[0](y)
        return y

# No projection: identity shortcut
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=0, in_decoder=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_task(in_planes, planes, stride)
        self.conv2 = nn.Sequential(nn.ReLU(True), conv_task(planes, planes, 1, second=1))
        self.shortcut = shortcut
        if self.shortcut == 1:
            if in_decoder:
                self.uppool = nn.Upsample(scale_factor=2)
            elif stride == 1:
                self.avgpool = nn.AvgPool2d(1)
            else:
                self.avgpool = nn.AvgPool2d(2)
        self.in_decoder = in_decoder

    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut == 1:
            if self.in_decoder:
               residual = self.uppool(residual)[:,::2,:,:]
            else:
                #  residual = self.avgpool(x)
                residual = nn.functional.adaptive_avg_pool2d(x, y.shape[2])
                residual = torch.cat((residual, residual*0),1)
        y += residual
        return y

class ResNet(nn.Module):
    def __init__(self, block, nblocks, num_classes=[10], final_pool=True,
                 mlp_depth=1, stride=2, color_channels=3, color_temp=1,
                 num_color_classes=313, normalize=False):
        super(ResNet, self).__init__()
        blocks = [block, block, block]
        factor = 1
        self.factor = factor
        self.in_planes = int(32*factor)
        self.pre_layers_conv = conv_task(color_channels,int(32*factor), 1)
        self.layer1 = self._make_layer(blocks[0], int(64*factor), nblocks[0], stride=stride)
        self.layer2 = self._make_layer(blocks[1], int(128*factor), nblocks[1], stride=stride)
        self.layer3 = self._make_layer(blocks[2], int(256*factor), nblocks[2], stride=stride)
        self.end_bns = nn.Sequential(nn.BatchNorm2d(int(256*factor)),nn.ReLU(True))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.final_pool = final_pool
        self.stride = stride
        self.normalize = normalize
        if normalize:
            self.l2norm = Normalize(2)
        linear_ins = int(256*factor) if final_pool else int(256*8*8*factor)
        self.linears = self._mlp_layer(int(linear_ins), num_classes, depth=mlp_depth)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _mlp_layer(self, linear_ins, num_classes, depth=1):
        layers = nn.ModuleList([])
        for layer_i in range(depth-1):
            layers.append(nn.Sequential(nn.Linear(linear_ins, linear_ins), nn.ReLU(True)))
        layers.append(nn.Linear(linear_ins, num_classes))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, nblocks, stride=1):
        shortcut = 0
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = 1
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks-1):
            layers.append(block(self.in_planes, planes))
            layers.append(nn.ReLU(True))
        layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, return_layer=4, rep_dim=256):
        if return_layer not in {1, 2, 3, 4}: raise NotImplementedError
        x = self.pre_layers_conv(x)
        x = self.layer1(x)
        if self.stride == 1: x = self.intermediate_pool(x)
        if return_layer==1: return x
        x = F.relu(x)
        x = self.layer2(x)
        if self.stride == 1: x = self.intermediate_pool(x)
        if return_layer==2: return x
        x = F.relu(x)
        x = self.layer3(x)
        if return_layer==3:
            num_channels = x.shape[1]
            resized_spatial_features = rep_dim / num_channels
            resized_spatial_dim = np.sqrt(resized_spatial_features)
            assert resized_spatial_dim == int(resized_spatial_dim)
            x = F.adaptive_avg_pool2d(x, int(resized_spatial_dim))
            # print(x.shape)
            # print(x.min())
            return x
        x = self.end_bns(x)
        if self.final_pool: x = self.avgpool(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linears(x)
        if self.normalize:
            x = self.l2norm(x)
        return x

class Generator(nn.Module):
    def __init__(self, bottleneck_dim, ngf=64):
        super(Generator, self).__init__()
        ngf = ngf
        nz = bottleneck_dim
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        # print(x.shape)
        return self.main(x)


def resnet26(num_classes=[10], blocks=BasicBlock, final_pool=True,
                mlp_depth=1, stride=2, color_channels=3,
                color_temp=1., num_color_classes=313,
                normalize=False):
    return  ResNet(blocks, [4,4,4],num_classes, final_pool=final_pool,
                   mlp_depth=mlp_depth, stride=stride,
                    color_channels=color_channels,
                    color_temp=color_temp,
                    num_color_classes=num_color_classes,
                    normalize=normalize)

class AE(nn.Module):
    def __init__(self, bottleneck_dim, image_dim=64, use_final_pool=True,
                  ngf=64):
        super(AE, self).__init__()
        if image_dim==64:
            self.encoder = resnet26(num_classes=bottleneck_dim, final_pool=use_final_pool)
            # print(self.encoder)
            self.decoder = Generator(bottleneck_dim=bottleneck_dim, ngf=ngf)

    def forward(self, x):
        code = self.encoder(x)
        # print(code.shape)
        return self.decoder(code)

class LinearFromRepModel(nn.Module):

    def __init__(self, feature_extractor_net, downsample_res=6, return_layer=3, num_classes=100, mode='', defined_rep_dim=0, mlp_depth=1):
        super(LinearFromRepModel, self).__init__()
        self.mode = mode
        try:
            factor = feature_extractor_net.factor if hasattr(feature_extractor_net, 'factor') else 1
        except:
            factor = feature_extractor_net.module.factor if hasattr(feature_extractor_net.module, 'factor') else 1
        if mode=='jigsaw':
            feature_extractor_net.shuffle = False
        else:
            # ae_factor = (8 ** 2) if mode=='ae' else 1
            ae_factor = (8 ** 2) if (not feature_extractor_net.final_pool)  else 1
            rep_dim = int(32 * (2**return_layer) * factor * ae_factor)
        if defined_rep_dim:
            rep_dim = defined_rep_dim
        self.return_layer = return_layer
        self.num_classes = num_classes
        self.linear_model = nn.Sequential(
                                            Flatten(),
                                            self._mlp_layer(rep_dim, num_classes, depth=mlp_depth),
                                        )
        self.feature_extractor = feature_extractor_net # nn.DataParallel(feature_extractor_net).cuda()
        self.rep_dim = rep_dim

    def _mlp_layer(self, linear_ins, num_classes, depth=1):
        layers = nn.ModuleList([])
        for layer_i in range(depth-1):
            layers.append(nn.Sequential(nn.Linear(linear_ins, linear_ins), nn.ReLU(True)))
        layers.append(nn.Linear(linear_ins, num_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.feature_extractor(x, return_layer=self.return_layer, rep_dim=self.rep_dim)
        return self.linear_model(features)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)



