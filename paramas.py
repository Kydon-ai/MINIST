# 超参数
EPOCH = 1  # 训练整批数据的次数
BATCH_SIZE = 50
LR = 0.001  # 学习率
# 首次使用请设置为True，下载数据集后设置为False
DOWNLOAD_MNIST = False  # 表示还没有下载数据集，如果数据集下载好了就写False
USE_DATA_TRAIN_NUM = 32 # 使用训练数据数量
USE_DATA_TEST_NUM = 20 # 使用测试数据数量
SAVE_MODULE_PATH = './output/' # 保存模型路径
SAVE_MODULE_NAME = 'cnn2.pkl' # 保存模型名称
SAVE_PIC_NAME = 'mnist_predictions.png' # 保存图片名称
