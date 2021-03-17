"""
    date:       2021/3/15 4:56 下午
    written by: neonleexiang
"""
import numpy as np
import os
import cv2 as cv
import tensorflow as tf


def psnr(img1, img2):
    diff = img1 - img2
    mse = np.mean(np.square(diff))
    return 10 * np.log10(255 * 255 / mse)


class Self_Defined_psnr_accuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.psnr_result = []
        # self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        # self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        # values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        # self.total.assign_add(tf.shape(y_true)[0])
        # self.count.assign_add(tf.reduce_sum(values))
        for t, p in zip(y_true, y_pred):
            self.psnr_result.append(psnr(t * 255., p * 255.))

    def result(self):
        return np.mean(self.psnr_result)
        # return self.count / self.total



if __name__ == '__main__':
    prefix = 'result/data_pred'
    dir_1 = '-img.png'
    dir_2 = '-pred.png'
    p_result = []
    for i in range(1, 10):
        img1_dir = os.path.join(prefix, str(i)+dir_1)
        img2_dir = os.path.join(prefix, str(i)+dir_2)
        img1 = cv.imread(img1_dir, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2_dir, cv.IMREAD_GRAYSCALE)
        p_result.append(psnr(img1, img2))

    print(np.mean(p_result))

    # dir = 'result/data_pred/1-img.png'
    # img1 = cv.imread(dir, cv.IMREAD_GRAYSCALE)
    # print(img1)
