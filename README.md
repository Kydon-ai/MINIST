# MINIST
pytorch+MINIST实现手写数字识别<br>
fork自[MINIST](https://github.com/jing-repo/MINIST)<br>

## 数据集
通过`torchvision.datasets.MNIST`自动下载到data文件夹<br>
首次运行程序请设置DOWNLOAD_MNIST为True，下载数据集后设置为False<br>
## 依赖
使用`pip freeze > requirements.txt`导出<br>
可直接使用`pip install -r requirements.txt`安装依赖<br>
> 如果cuda版本过低，请于https://pytorch.org/get-started/locally/自行选择合适的pytorch版本。如果大版本不同，请自行修改部分语法！
