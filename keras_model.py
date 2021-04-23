"""
    date:       2021/3/19 11:12 上午
    written by: neonleexiang
"""
import os
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from keras.layers.core import Activation


class SRCNN:
    def __init__(self, image_size, c_dim, is_training, learning_rate=1e-4, batch_size=128, epochs=1500):
        """

        :param image_size: 图像大小
        :param c_dim: 图像图层维度
        :param is_training:
        :param learning_rate:
        :param batch_size:
        :param epochs:
        """
        self.image_size = image_size
        self.c_dim = c_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.is_training = is_training
        if self.is_training:
            self.model = self.build_model()
        else:
            self.model = self.load()

    def build_model(self):
        """
        keras 的 model 使用 Sequential 作为模型结构，可以通过 add的方式添加每层的结构
        最后使用 compile 配合 optimizer loss metrics 【评估标注】

        :return:  model
        """
        model = Sequential()
        # input_size 为64， 9*9 -> 1*1 -> 5*5
        model.add(Conv2D(64, 9, padding='same', input_shape=(self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 1, padding='same'))
        model.add(Activation('relu'))
        # output size = c_dim
        model.add(Conv2D(self.c_dim, 5, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        return model

    def train(self, X_train, Y_train):
        """
        keras 使用 model.fit 作为模型输入进行训练，同时传入 batch_size, epochs verbose validation_split

        :param X_train:
        :param Y_train:
        :return:
        """
        """
            verbose：日志显示
            verbose = 0 为不在标准输出流输出日志信息
            verbose = 1 为输出进度条记录
            verbose = 2 为每个epoch输出一行记录
            注意： 默认为 1
        """
        """
            evaluate 中的 verbose

            verbose：日志显示
            verbose = 0 为不在标准输出流输出日志信息
            verbose = 1 为输出进度条记录
            注意： 只能取 0 和 1；默认为 1
        """
        history = self.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                                 validation_split=0.1)
        if self.is_training:
            self.save()
        return history

    def process(self, inputs):
        """
        predict

        :param inputs:
        :return:
        """
        predicted = self.model.predict(inputs)
        return predicted

    def load(self):
        """
        load data
        :return:
        """
        weight_filename = 'srcnn_weight.hdf5'
        model = self.build_model()
        model.load_weights(os.path.join('./model/', weight_filename))
        return model

    def save(self):
        """
        save data
        :return:
        """
        json_string = self.model.to_json()
        if not os.path.exists('model'):
            os.mkdir('model')
        open(os.path.join('model/', 'srcnn_model.json'), 'w').write(json_string)
        self.model.save_weights(os.path.join('model/', 'srcnn_weight.hdf5'))
        return json_string
