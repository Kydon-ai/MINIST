import torch 
import torchvision
import cv2
import os
from paramas import USE_DATA_TEST_NUM, SAVE_MODULE_PATH, SAVE_MODULE_NAME, SAVE_PIC_NAME

def verification(cnn,test_x):
    # 旧版本代码
    # cnn.load_state_dict(torch.load('cnn2.pkl'))

    # 新版本代码：添加map_location确保兼容性
    cnn.load_state_dict(torch.load(os.path.join(SAVE_MODULE_PATH, SAVE_MODULE_NAME), map_location=torch.device('cpu')))
    cnn.eval()
    # print 10 predictions from test data
    inputs = test_x[:USE_DATA_TEST_NUM]  # 测试USE_DATA_TEST_NUM个数据
    test_output = cnn(inputs)
    # 旧版本代码
    # pred_y = torch.max(test_output, 1)[1].data.numpy()
    # 新版本代码：直接使用numpy()
    pred_y = torch.max(test_output, 1)[1].numpy()
    print(pred_y, 'prediction number')  # 打印识别后的数字
    # print(test_y[:10].numpy(), 'real number')

    img = torchvision.utils.make_grid(inputs)
    img = img.numpy().transpose(1, 2, 0)

    # 下面三行为改变图片的亮度
    # std = [0.5, 0.5, 0.5]
    # mean = [0.5, 0.5, 0.5]
    # img = img * std + mean

    # 在服务器环境中直接保存图像
    img = img * 255  # 转换回0-255范围以便保存
    img = img.astype('uint8')  # 转换为8位无符号整数
    cv2.imwrite(os.path.join(SAVE_MODULE_PATH, SAVE_PIC_NAME), img)  # 保存图像
    print(f"图像已保存为 {os.path.join(SAVE_MODULE_PATH, SAVE_PIC_NAME)}")

