import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.encode_dim = int((self.config.H/16)*(self.config.W/16)*32)

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 2, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 2, stride=2)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 2, stride=2)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv15 = nn.ConvTranspose2d(128, 128 , 2, stride=2)
        self.bn15 = nn.BatchNorm2d(128)
        self.conv16 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn16 = nn.BatchNorm2d(128)
        self.conv17 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn17 = nn.BatchNorm2d(64)
        self.conv18 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.bn18 = nn.BatchNorm2d(64)
        self.conv19 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn19 = nn.BatchNorm2d(64)
        self.conv20 = nn.Conv2d(64, 1, 3, padding=1)
        self.bn20 = nn.BatchNorm2d(1)


    def forward(self, x):
        x1 = F.elu(self.bn1(self.conv1(x)))
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.elu(self.bn3(self.conv3(x2)))
        x4 = F.elu(self.bn4(self.conv4(x3)))
        x5 = F.elu(self.bn5(self.conv5(x4)))
        x6 = F.elu(self.bn6(self.conv6(x5)))
        x7 = F.elu(self.bn7(self.conv7(x6)))
        x8 = F.elu(self.bn8(self.conv8(x7)))
        x9 = F.relu(self.bn9(self.conv9(x8)))
        x9 = torch.cat((x9, x7), dim=1)
        x10 = F.relu(self.bn10(self.conv10(x9)))
        x11 = F.relu(self.bn11(self.conv11(x10)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x12 = torch.cat((x12, x5), dim=1)
        x13 = F.relu(self.bn13(self.conv13(x12)))
        x14 = F.relu(self.bn14(self.conv14(x13)))
        x15 = F.relu(self.bn15(self.conv15(x14)))
        x15 = torch.cat((x15, x3), dim=1)
        x16 = F.relu(self.bn16(self.conv16(x15)))
        x17 = F.relu(self.bn17(self.conv17(x16)))
        x18 = F.relu(self.bn18(self.conv18(x17)))
        x18 = torch.cat((x18, x1), dim=1)
        x19 = F.relu(self.bn19(self.conv19(x18)))
        fx = F.sigmoid(self.bn20(self.conv20(x19)))

        return fx

class Discriminator(nn.Module):#DCGAN判别器
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        # Spectral Normalization谱归一化
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 64, 4, padding=1, stride=2, bias=False))
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, padding=1, stride=2, bias=False))
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, padding=1, stride=2, bias=False))
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, padding=1, stride=2, bias=False))
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, stride=1, bias=False))
        self.act5 = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.act1(self.conv1(x))
        conv2 = self.act2(self.conv2(conv1))
        conv3 = self.act3(self.conv3(conv2))
        conv4 = self.act4(self.conv4(conv3))
        output = self.act5(self.conv5(conv4))
        return output