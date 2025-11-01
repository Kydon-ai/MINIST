from paramas import EPOCH, LR
import torch
import torch.nn as nn
import os
from paramas import SAVE_MODULE_PATH, SAVE_MODULE_NAME
def train(cnn,train_loader,test_x,test_y):
    # 训练
    # 把x和y 都放入Variable中，然后放入cnn中计算output，最后再计算误差
    # 优化器选择Adam
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    # 损失函数
    loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted
    # 检查模型文件是否已存在
    model_path = os.path.join(SAVE_MODULE_PATH, SAVE_MODULE_NAME)
    if os.path.exists(model_path):
        print(f"模型文件 {model_path} 已存在，跳过训练")
        return
    
    print("开始训练...")
    # 确保有train_loader和test_x变量可用
    # 这里假设这些变量是从其他地方导入的
    
    # 开始训练
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
            output = cnn(b_x)  # 先将数据放到cnn中计算output
            loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
            optimizer.zero_grad()  # 清除之前学到的梯度的参数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 应用梯度

            if step % 50 == 0:
                test_output = cnn(test_x)
            # 旧版本代码
                # pred_y = torch.max(test_output, 1)[1].data.numpy()
                # accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
                
                # 新版本代码
                pred_y = torch.max(test_output, 1)[1].numpy()
                accuracy = float((pred_y == test_y.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)

    torch.save(cnn.state_dict(), model_path)#保存模型
    print(f"模型已保存到 {model_path}")