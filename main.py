from paramas import EPOCH, BATCH_SIZE, LR, DOWNLOAD_MNIST, USE_DATA_TEST_NUM
# 训练+测试
import torch
import torch.utils.data as Data
import torchvision
from CNN import CNN

cnn = CNN()
print(cnn)

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同


if __name__ == '__main__':
    # 下载mnist手写数据集
    train_data = torchvision.datasets.MNIST(
        root='./data/',  # 保存或提取的位置  会放在当前文件夹中
        train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
        transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
        download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
    )

    test_data = torchvision.datasets.MNIST(
        root='./data/',
        train=False,  # 表明是测试集
        transform=torchvision.transforms.ToTensor()  # 测试集也需要应用同样的转换
    )

    # 批训练 50个samples， 1  channel，28x28 (50,1,28,28)
    # Torch中的DataLoader是用来包装数据的工具，它能帮我们有效迭代数据，这样就可以进行批训练
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True  # 是否打乱数据，一般都打乱
    )

    # 进行测试
    # 为节约时间，测试时只测试前USE_DATA_TEST_NUM个
    #
    # 旧版本代码
    # test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
    # # torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
    # # 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
    # test_y = test_data.test_labels[:2000]

    # 新版本代码：使用最新的API访问数据
    test_x = test_data.data[:USE_DATA_TEST_NUM].unsqueeze(1).float() / 255.0
    # .data替代了.train_data，unsqueeze替代了torch.unsqueeze，float()替代了.type(torch.FloatTensor)
    test_y = test_data.targets[:USE_DATA_TEST_NUM]  # .targets替代了.test_labels

    from state.pretrain import train
    train(cnn,train_loader,test_x,test_y)

    from state.verification import verification
    verification(cnn,test_x)
