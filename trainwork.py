import torch
import torch.nn as nn
from prework import TrainDataset
from network import Generator, Discriminator
# from networkunet import Generator, Discriminator
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time
# 过滤警告信息
import warnings
warnings.filterwarnings("ignore")

class CNNConfig():
    dataset_dir = './dataset'
    test_dir = './test'
    H = 64
    W = 64
    ClASSES = None
    EPOCH = 40
    BATCH_SIZE = 1
    LR = 0.0001  # learning rate

config = CNNConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
dataset = TrainDataset(config.dataset_dir,transform=data_transform)
lenlabel = dataset.lenlabel
config.ClASSES = lenlabel + 1
#使用DataLoader加载数据
train_loader = DataLoader(dataset,batch_size=config.BATCH_SIZE,num_workers=0,shuffle=True)
total_step = len(train_loader)

# 加载模型
generator = Generator(config)
generator.to(device)
discriminator = Discriminator(config)
discriminator.to(device)

g_loss = nn.MSELoss()
d_loss = nn.BCELoss()  # 是单目标二分类交叉熵函数
g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.LR)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.LR*0.1)

writer = SummaryWriter()#tensorboard
global_step = 0
for epoch in range(config.EPOCH):
    since = time.time()
    for i, (x_batch, y_batch, labels) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        labels = labels.to(device)
        global_step += 1

        # 训练判别器D
        G_model = generator(x_batch)
        y_fake = G_model.detach()  # 分离生成器参数，原参数固定
        D_fake = discriminator(y_fake)
        D_real = discriminator(y_batch)

        D_loss_real = d_loss(D_real, torch.ones(x_batch.shape[0]).to(device))
        D_loss_fake = d_loss(D_fake, torch.zeros(x_batch.shape[0]).to(device))
        D_loss = D_loss_real + D_loss_fake
        d_optimizer.zero_grad()  # 先清空所有参数的梯度缓存，否则会在上面累加
        D_loss.backward()  # 计算反向传播
        d_optimizer.step()  # 更新梯度

        # 训练生成器G
        D_fake = discriminator(G_model)
        G_loss_gan = d_loss(D_fake, torch.ones(x_batch.shape[0]).to(device))
        G_loss_MSE = g_loss(G_model, y_batch)
        G_loss = G_loss_MSE + G_loss_gan * 0.0001
        g_optimizer.zero_grad()
        G_loss.backward()
        g_optimizer.step()

        G_loss_gan_value = G_loss_gan.item()
        G_loss_value = G_loss.item()  # tensor转numpy
        D_loss_value = D_loss.item()

        writer.add_scalar('G_loss', G_loss_value, global_step=global_step)
        writer.add_scalar('G_loss_gan', G_loss_gan_value, global_step=global_step)
        writer.add_scalar('D_loss', D_loss_value, global_step=global_step)

        if (i + 1) % 50 == 0:
            print('Epoch[{}/{}],Step[{}/{}],G_loss:{:.8f},G_loss_gan:{:.8f},D_loss:{:.8f},Global_step:{}'
                  .format(epoch + 1, config.EPOCH, i + 1, total_step, G_loss_value, G_loss_gan_value,
                          D_loss_value, global_step))

    now = time.time()
    time_elapsed = now - since
    print('Training complete in {:.0f}m {:.3f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if (epoch + 1) % 5 == 0:
        torch.save(discriminator.state_dict(), './logs/discriminator-' + str(epoch + 1) + '.pth')
        torch.save(generator.state_dict(), './logs/generator-' + str(epoch + 1) + '.pth')

writer.close()

