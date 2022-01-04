import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.conv_1 = nn.Conv2d(1, 32, 11, padding=5)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(32)
        self.conv_3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(32)
        self.conv_4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(32)
        self.conv_5 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(32)
        self.conv_6 = nn.Conv2d(32, 1, 3, padding=1)
        self.bn_6 = nn.BatchNorm2d(1)

        self.conv1 = nn.Conv2d(2, 32, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9_up = nn.Upsample(size=(int(self.config.H/4), int(self.config.W/4)))
        self.conv9 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(64)
        self.conv11_up = nn.Upsample(size=(int(self.config.H/2), int(self.config.W/2)))
        self.conv11 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(32)
        self.conv13_up = nn.Upsample(size=(int(self.config.H), int(self.config.W)))
        self.conv13 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(16)
        self.conv14 = nn.Conv2d(16, 8, 3, padding=1)
        self.bn14 = nn.BatchNorm2d(8)
        self.conv15 = nn.Conv2d(8, 1 ,3, padding=1)
        self.bn15 = nn.BatchNorm2d(1)

    def forward(self, X):
        y = X
        y = F.relu(self.bn_1(self.conv_1(y)))
        y = F.relu(self.bn_2(self.conv_2(y)))
        y = F.relu(self.bn_3(self.conv_3(y)))
        y = F.relu(self.bn_4(self.conv_4(y)))
        y = F.relu(self.bn_5(self.conv_5(y)))
        y = F.sigmoid(self.bn_6(self.conv_6(y)))

        x = torch.cat((X, y), dim=1)#tensor拼接
        x = F.relu(self.bn1(self.conv1(F.pad(x, pad=(1, 2, 1, 2)))))
        x = F.relu(self.bn2(self.conv2(F.pad(x, pad=(0, 1, 0, 1)))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(F.pad(x, pad=(0, 1, 0, 1)))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.conv9_up(x))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.conv11_up(x))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.conv13_up(x))
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.tanh(self.bn15(self.conv15(x)))
        x = torch.add(x, y)
        a = torch.zeros_like(x)
        b = torch.ones_like(x)
        fx = torch.max(a, torch.min(b, x))
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