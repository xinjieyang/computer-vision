from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
# 过滤警告信息
import warnings
warnings.filterwarnings("ignore")

class TrainDataset(Dataset):  # 继承Dataset
    def __init__(self, path_dir, transform=None):  # 初始化一些属性
        self.path_dir = path_dir  # 文件路径,
        self.transform = transform  # 对图形进行处理，如标准化、截取、转换等
        self.datainput = os.path.join(self.path_dir, 'input')
        self.datatarget = os.path.join(self.path_dir, 'target')
        self.label = os.listdir(self.datainput)  # 把路径下的所有文件放在一个列表中
        self.lenlabel = len(self.label)
        self.inimg_dir, self.tarimg_dir, self.label_ = self.get_img_dir()

    def get_img_dir(self):
        indir = []
        tardir = []
        label = []
        for i in range(self.lenlabel):
            next_indir = os.path.join(self.datainput, self.label[i])
            next_targetdir = os.path.join(self.datatarget, self.label[i])
            for file in os.listdir(next_indir):
                indir.append(os.path.join(next_indir, file))
                tardir.append(os.path.join(next_targetdir, file))
                label.append(i)
        label = [int(i) for i in label]
        return indir, tardir, label

    def __len__(self):  # 返回整个数据集的大小
        return len(self.inimg_dir)

    def __getitem__(self, index):  # 根据索引index返回图像及标签
        inimg_path = self.inimg_dir[index]  # 根据索引获取图像文件名称
        tarimg_path = self.tarimg_dir[index]
        labelimg = self.label_[index]

        inimg = Image.open(inimg_path).convert('L')  # 读取灰度图像
        tarimg = Image.open(tarimg_path).convert('L')

        if self.transform is not None:
            inimg = self.transform(inimg)
            tarimg = self.transform(tarimg)
        return inimg, tarimg, labelimg

class TestDataset(Dataset):  # 继承Dataset
    def __init__(self, path_dir, transform=None):  # 初始化一些属性
        self.path_dir = path_dir  # 文件路径,
        self.transform = transform  # 对图形进行处理，如标准化、截取、转换等
        self.datainput = os.path.join(self.path_dir, 'input')
        self.datatarget = os.path.join(self.path_dir, 'target')
        self.inimg_dir, self.tarimg_dir = self.get_img_dir()

    def get_img_dir(self):
        indir = []
        tardir = []
        for file in os.listdir(self.datainput):
            indir.append(os.path.join(self.datainput, file))
            tardir.append(os.path.join(self.datatarget, file))
        return indir, tardir

    def __len__(self):  # 返回整个数据集的大小
        return len(self.inimg_dir)

    def __getitem__(self, index):  # 根据索引index返回图像及标签
        inimg_path = self.inimg_dir[index]  # 根据索引获取图像文件名称
        tarimg_path = self.tarimg_dir[index]

        inimg = Image.open(inimg_path).convert('L')  # 读取灰度图像
        tarimg = Image.open(tarimg_path).convert('L')

        if self.transform is not None:
            inimg = self.transform(inimg)
            tarimg = self.transform(tarimg)
        return inimg, tarimg

