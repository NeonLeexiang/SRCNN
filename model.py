"""
    date:       2021/3/9 3:42 下午
    written by: neonleexiang
"""
import tensorflow as tf


# building SRCNN using tensorflow 2.0
class SRCNN(tf.keras.Model):
    """
        according to the tensorflow document:

        class MyModel(tf.keras.Model):
            def __init__(self):
                super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
                # 此处添加初始化代码（包含 call 方法中会用到的层），例如
                # layer1 = tf.keras.layers.BuiltInLayer(...)
                # layer2 = MyCustomLayer(...)

            def call(self, input):
                # 此处添加模型调用的代码（处理输入并返回输出），例如
                # x = layer1(input)
                # output = layer2(x)
                return output

            # 还可以添加自定义的方法

        example:

        class CNN(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.conv1 = tf.keras.layers.Conv2D(
                    filters=32,             # 卷积层神经元（卷积核）数目
                    kernel_size=[5, 5],     # 感受野大小
                    padding='same',         # padding策略（vaild 或 same）
                    activation=tf.nn.relu   # 激活函数
                )
                self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
                self.conv2 = tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=[5, 5],
                    padding='same',
                    activation=tf.nn.relu
                )
                self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
                self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
                self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
                self.dense2 = tf.keras.layers.Dense(units=10)

            def call(self, inputs):
                x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
                x = self.pool1(x)                       # [batch_size, 14, 14, 32]
                x = self.conv2(x)                       # [batch_size, 14, 14, 64]
                x = self.pool2(x)                       # [batch_size, 7, 7, 64]
                x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
                x = self.dense1(x)                      # [batch_size, 1024]
                x = self.dense2(x)                      # [batch_size, 10]
                output = tf.nn.softmax(x)
                return output

    """
    def __init__(self):
        super(SRCNN, self).__init__()
        """
            according to the paper, the structure of the model is 3 layers
            for every layer: 9*9 -> 1*1 -> 5*5
            and the filters is 64 -> 32 -> 1
            and we use the relu activation
            for padding, we set 'same'
            but if we do not want to have the same output img size we can set 'valid'
        """
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[9, 9],
            padding='same',
            activation=tf.nn.relu,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[1, 1],
            padding='same',
            activation=tf.nn.relu,
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu,
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        output = x
        return output

