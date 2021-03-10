"""
    date:       2021/3/9 3:42 下午
    written by: neonleexiang
"""
import tensorflow as tf


class SRCNN(tf.keras.Model):
    def __init__(self):
        super(SRCNN, self).__init__()
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
            filters=3,
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

