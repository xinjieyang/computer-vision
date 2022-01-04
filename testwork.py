import torch
from prework import TestDataset
from network import Generator
# from networkunet import Generator
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import cv2

class CNNConfig():
    dataset_dir = './dataset'
    test_dir = './test'
    H = 64
    W = 64
    ClASSES = None
    EPOCH = 40
    BATCH_SIZE = 8
    LR = 0.0001  # learning rate

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = CNNConfig()
generator = Generator(config)
#载入GPU训练的预参数
generator.load_state_dict(torch.load('./logs/generator-4.pth'))
#将模型设为评估模式，在模型中禁用dropout或者batch normalization层
generator.eval()

data_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
dataset = TestDataset(config.test_dir,transform=data_transform)
test_loader = DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=None)
# 在模型中禁用autograd功能，加快计算
with torch.no_grad():
    for i, (x_batch, y_batch) in enumerate(test_loader):
        prediction = generator(x_batch)

        # 显示图像
        # imshow(utils.make_grid(x_batch))
        # imshow(utils.make_grid(y_batch))
        imshow(utils.make_grid(prediction))

        # 去掉1的维度,从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
        prediction = prediction.squeeze().mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).permute(1, 2, 0).numpy()
        for j in range(prediction.shape[2]):
            output = prediction[0:config.H, 0:config.W, j:j + 1]
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)# RGB转BRG
            cv2.imwrite(config.test_dir+'/output/%04d.jpg'%(config.BATCH_SIZE*i+j), output)




