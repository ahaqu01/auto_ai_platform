# 导入必要的库
import errno

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import transforms, datasets, models, utils
from torchsummary import summary  # 可视化训练过程
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from PIL import Image
import torch.nn.functional as F
import torchvision.models as models

# 分为为train, val, test定义transform
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=300, scale=(0.8, 1.1)),  # 功能：随机长宽比裁剪原始图片, 表示随机crop出来的图片会在的0.08倍至1.1倍之间
        transforms.RandomRotation(degrees=10),  # 功能：根据degrees随机旋转一定角度, 则表示在（-10，+10）度之间随机旋转
        transforms.ColorJitter(0.4, 0.4, 0.4),  # 功能：修改亮度、对比度和饱和度
        transforms.RandomHorizontalFlip(),  # 功能：水平翻转
        transforms.CenterCrop(size=256),  # 功能：根据给定的size从中心裁剪，size - 若为sequence,则为(h,w)，若为int，则(size,size)
        transforms.ToTensor(),  # numpy --> tensor
        # 功能：对数据按通道进行标准化（RGB），即先减均值，再除以标准差
        transforms.Normalize([0.485, 0.456, 0.406],  # mean
                             [0.229, 0.224, 0.225])  # std
    ]),

    'val': transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # mean
                             [0.229, 0.224, 0.225])  # std
    ]),

    'test': transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # mean
                             [0.229, 0.224, 0.225])  # std
    ])
}
import torchvision
def load_datasets(train_dir, val_dir, test_dir):
    """
    load datasets, and transform with datasets .
    :param train_dir: the train dataset direction.
    :param val_dir: the val dataset direction.
    :param test_dir: the test dateset direction.
    :return:
    """
    try:
        # 从文件中读取数据
        datasets = {
            'train': torchvision.datasets.ImageFolder(train_dir, transform=image_transforms['train']),  # 读取train中的数据集，并transform
            'val': torchvision.datasets.ImageFolder(val_dir, transform=image_transforms['val']),  # 读取val中的数据集，并transform
            'test': torchvision.datasets.ImageFolder(test_dir, transform=image_transforms['test'])  # 读取test中的数据集，并transform
        }
    except Exception as e:
        return errno
    return datasets


def dataloader_iterator(datasets, batch_size):
    """
    加载数据后，创建一个迭代器，按批次读取数据
    :param datasets: 数据集
    :param batch_size:每批次读取的数据量
    :return: 返回加载数据的迭代器
    """
    try:
        # DataLoader : 创建iterator, 按批读取数据
        dataloaders = {
            'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True),  # 训练集
            'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=True),  # 验证集
            'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=True)  # 测试集
        }
    except Exception as e:
        return errno
    return dataloaders

def label_k_value(datasets):
    """
    创建标签的键对值
    :param datasets:数据集
    :return: 标签的键对值
    """
    try:
        # 创建label的键值对
        LABEL = dict((v, k) for k, v in datasets['train'].class_to_idx.items())
    except Exception as e:
        return errno
    return LABEL

# 导入SummaryWriter
from torch.utils.tensorboard import SummaryWriter

# SummaryWriter() 向事件文件写入事件和概要

# 定义日志路径
log_path = os.path.abspath('.')  +'/logdir/'

# 定义函数：获取tensorboard writer
def tb_writer():
    timestr = time.strftime("%Y%m%d_%H%M%S")  # 时间格式
    writer = SummaryWriter(log_path + timestr)  # 写入日志
    return writer

# 记录错误分类的图片
def misclassified_images(pred, writer, target, images, LABEL, epoch, count=10):
    misclassified = (pred != target.data)  # 判断是否一致
    for index, image_tensor in enumerate(images[misclassified][:count]):
        img_name = 'Epoch:{}-->Predict:{}-->Actual:{}'.format(epoch, LABEL[pred[misclassified].tolist()[index]],
                                                              LABEL[target.data[misclassified].tolist()[index]])
        writer.add_image(img_name, image_tensor, epoch)

# 自定义池化层

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        size = size or (1, 1)  # kernel大小
        # 自适应算法能够自动帮助我们计算核的大小和每次移动的步长。
        self.avgPooling = nn.AdaptiveAvgPool2d(size)  # 自适应平均池化
        self.maxPooling = nn.AdaptiveMaxPool2d(size)  # 最大池化

    def forward(self, x):
        # 拼接avg和max
        return torch.cat([self.maxPooling(x), self.avgPooling(x)], dim=1)

# 定义训练函数
def train_val(model, device, train_loader, val_loader, optimizer, criterion, epoch, writer):
    """
    定义训练函数
    :param model:模型
    :param device: 运行模型的设备，如CPU、GPU
    :param train_loader: 训练集
    :param val_loader: 验证集
    :param optimizer: 优化器
    :param criterion: 损失函数
    :param epoch: 训练次数
    :param writer: tensorboard记录训练
    :return:
            train_loss：平均训练损失
            val_loss：平均验证损失
            val_acc：平均准确率
    """
    model.train()
    total_loss = 0.0
    val_loss = 0.0
    val_acc = 0
    print(f"in train_val: train_loader.dataset = {train_loader.dataset}")
    for batch_id, (images, labels) in enumerate(train_loader):# enumerate():Python内置函数,用于将一个可遍历的数据对象（如列表、元组、或字符串）组合为一个索引序列，一般用在for循环当中。
        # 部署到device上
        images, labels = images.to(device), labels.to(device)
        # 梯度置0
        optimizer.zero_grad()
        # 模型输出
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
         # 更新参数
        optimizer.step()
        # 累计损失
        total_loss += loss.item() * images.size(0)

    # 平均训练损失
    train_loss = total_loss / len(train_loader.dataset)
    # 写入到writer中
    writer.add_scalar('Training Loss', train_loss, epoch)
    # 写入到磁盘
    writer.flush()

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 前向传播输出
            loss = criterion(outputs, labels)  # 损失
            val_loss += loss.item() * images.size(0)  # 累计损失
            _, pred = torch.max(outputs, dim=1)  # 获取最大概率的索引
            correct = pred.eq(labels.view_as(pred))  # 返回：tensor([ True,False,True,...,False])
            accuracy = torch.mean(correct.type(torch.FloatTensor))  # 准确率
            val_acc += accuracy.item() * images.size(0)  # 累计准确率
        # 平均验证损失
        val_loss = val_loss / len(val_loader.dataset)
        # 平均准确率
        val_acc = val_acc / len(val_loader.dataset)

    return train_loss, val_loss, val_acc

def get_model():
    """
    迁移学习：获取预训练模型，并替换池化层和全连接层
    :return: 替换掉池化层和全连接层的预训练模型
    """
    # 获取欲训练模型 restnet50
    model = models.resnet50(pretrained=True)
    # model = models.resnet152(pretrained=True)
    #  model = models.yolov3(pretrained=True)
    # 冻结模型参数
    for param in model.parameters():
        param.requires_grad = False
        # 替换最后2层：池化层和全连接层
    # 池化层
    model.avgpool = AdaptiveConcatPool2d()
    # 全连接层
    model.fc = nn.Sequential(
        nn.Flatten(),  # 拉平
        nn.BatchNorm1d(4096),  # 加速神经网络的收敛过程，提高训练过程中的稳定性
        nn.Dropout(0.5),  # 丢掉部分神经元
        nn.Linear(4096, 512),  # 全连接层
        nn.ReLU(),  # 激活函数
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 2),  # 2个输出
        nn.LogSoftmax(dim=1)  # 损失函数：将input转换成概率分布的形式，输出2个概率
    )
    return model


def test(model, device, test_loader, criterion, epoch, writer):
    """
    定义测试函数
    :param model: 模型
    :param device: 运行模型的设备
    :param test_loader: 测试集
    :param criterion: 损失函数
    :param epoch: 训练测试迭代次数
    :param writer: 记录训练测试
    :return:
            total_loss：累计损失
            accuracy：正确率
    """
    model.eval()
    total_loss = 0.0
    correct = 0.0  # 正确数
    with torch.no_grad():
        for batch_id, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            # 输出
            outputs = model(images)
            # 损失
            loss = criterion(outputs, labels)
            # 累计损失
            total_loss += loss.item()
            # 获取预测概率最大值的索引
            _, predicted = torch.max(outputs, dim=1)
            # 累计正确预测的数
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
            # 错误分类的图片
            misclassified_images(predicted, writer, labels, images, outputs, epoch)
        # 平均损失
        avg_loss = total_loss / len(test_loader.dataset)
        # 计算正确率
        accuracy = 100 * correct / len(test_loader.dataset)
        # 将test的结果写入write
        writer.add_scalar("Test Loss", total_loss, epoch)
        writer.add_scalar("Accuracy", accuracy, epoch)
        writer.flush()
        return total_loss, accuracy


def train_epochs(model, device, dataloaders, criterion, optimizer, epochs, writer):
    """
    定义训练流程函数
    :param model: 模型
    :param device: 运行模型使用的设备，CPU、GPU
    :param dataloaders: 数据集
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param epochs: 训练的次数
    :param writer: tensorboard记录训练
    :return:
    """
    # last_train_loss =0.0
    # last_val_loss = 0.0
    # last_val_acc = 0.0
    # last_test_loss = 0.0
    # last_test_acc = 0.0

    # 输出信息
    print(
        "{0:>15} | {1:>15}    | {2:>15}    | {3:>15} | {4:>15}  | {5:>15}".format('Epoch', 'Train Loss', 'val_loss', 'val_acc',
                                                                           'Test Loss', 'Test_acc'))
    # 初始最小的损失
    best_loss = np.inf
    # 开始训练、测试
    for epoch in range(epochs):
        # 训练，return: loss
        train_loss, val_loss, val_acc = train_val(model, device, dataloaders['train'], dataloaders['val'], optimizer,
                                                  criterion, epoch, writer)
        # 测试，return: loss + accuracy
        test_loss, test_acc = test(model, device, dataloaders['test'], criterion, epoch, writer)
        # 判断损失是否最小
        if test_loss < best_loss:
            best_loss = test_loss  # 保存最小损失
            # 保存模型
            torch.save(model.state_dict(), 'model.pth')
        # 输出结果
        print("{0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}".format(epoch, train_loss, val_loss, val_acc,
                                                                                 test_loss, test_acc))
        writer.flush()
        print(f'epoch = {epoch}, epochs = {epochs} ++++++++++++++++')
        if epoch == epochs - 1:
            last_train_loss = train_loss
            last_val_loss = val_loss
            last_val_acc = val_acc
            last_test_loss = test_loss
            last_test_acc = test_acc

    # 加载模型
    model.load_state_dict(torch.load('model.pth'))
    return last_train_loss, last_val_loss, last_val_acc, last_test_loss, last_test_acc, model

def train_two_caregory(train_dir, val_dir, tetst_dir, batch_size, epochs):
    """
    一定训练流程
    :return:
    """
    print(f' train_dir = {train_dir}, val_dir = {val_dir}, tetst_dir = {tetst_dir}')
    # 是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device.type)
    # 模型部署到device
    model = get_model().to(device)
    # 损失函数
    criterion = nn.NLLLoss()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    writer = tb_writer()
    datasets = load_datasets(train_dir, val_dir, tetst_dir)
    dataloaders = dataloader_iterator(datasets, batch_size)
    last_train_loss, last_val_loss, last_val_acc, last_test_loss, last_test_acc, model = \
        train_epochs(model, device, dataloaders, criterion, optimizer, epochs, writer)
    print("in train_two_caregory: {0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}".format(epochs,
                                last_train_loss, last_val_loss, last_val_acc,last_test_loss, last_test_acc))
    writer.close()

    return last_train_loss, last_val_loss, last_val_acc, last_test_loss, last_test_acc

def resnet_two_category_transfer_learn(train_name, val_name, test_name):
    abs_path = os.path.abspath('.')
    print(abs_path)
    current_path = abs_path + '/dataset/'
    train_dir = current_path + train_name + '/'
    val_dir = current_path + val_name + '/'
    test_dir = current_path + test_name + '/'
    EPOCHS = 1
    BARCH_SIZE = 128
    last_train_loss, last_val_loss, last_val_acc, last_test_loss, last_test_acc, model = \
        train_two_caregory(train_dir,val_dir, test_dir, BARCH_SIZE, EPOCHS)

    return last_train_loss, last_val_loss, last_val_acc, last_test_loss, last_test_acc, model


# 定义transform
def image_transformer(image_data):
    print('in image_transformer()')
    # 变换操作定义
    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # mean
                             [0.229, 0.224, 0.225])  # std
    ])
    print('before Image open')
    # 读取图片
    # image = Image.open(io.BytesIO(image_data)) # 读取图片, BytesIO实现了在内存中读写bytes
    image = Image.open(image_data)  # 读取图片, BytesIO实现了在内存中读写bytes
    print(f'after Image open')
    trans = transform(image).unsqueeze(0)
    print("trans shape : ", trans.shape)
    return trans


def predict_two_category_by_model(model, img_path, normal, off_normal):
    """
    通过自动化迁移学习训练的模型，进行二分类预测
    :param model: 迁移学习得到的模型
    :param img_path: 要预测的图像
    :return:
    """
    print(f"abs_path = {os.path.abspath('.')}")
    # 读取图片
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("before image_transforms")
    image = image_transformer(img_path)

    print("test image shape:", image.shape)
    # 加载模型
    pred = model(image.to(device))
    print("output : ", pred)
    # 计算概率
    proba = F.softmax(pred, dim=1)
    print("proba : ", proba)
    print("proba.detach()", proba.detach())
    print("value = ", proba.detach().cpu().numpy().flatten())
    if proba.detach().cpu().numpy().flatten()[0] > proba.detach().cpu().numpy().flatten()[1]:
        value = normal
    else:
        value = off_normal

    print(value)
    return value  # flatten() : 返回一个一维数组




# if __name__== '__main__':
#     resnet_two_category_transfer_learn('train', 'val', 'test')

