"""
    date:       2021/3/20 3:37 下午
    written by: neonleexiang
"""
from torch import nn


class pytorch_SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(pytorch_SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(9, 9), padding=(9//2, 9//2))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=(5, 5), padding=(5//2, 5//2))
        """
        inplace=True
        对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量。
        """
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x



