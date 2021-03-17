"""
    date:       2021/3/15 5:50 下午
    written by: neonleexiang
"""
import tensorflow as tf
import cv2 as cv
import numpy as np
from data_preprocessing import SRCNNLoader


batch_size = 4

model = tf.saved_model.load('saved/saved/1')
print(model)
data_loader = SRCNNLoader()
num_batches = int(data_loader.num_test_data // batch_size)

print(data_loader.test_data.shape)

i = 0

for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    data = data_loader.test_data[start_index: end_index]
    mod_size = len(data)
    data = tf.cast(data, dtype=tf.float32)
    print(data.shape)
    print(data.dtype)
    y_pred = model(data)
    y_true = data_loader.test_label[start_index: end_index]
    for t, p in zip(y_true, y_pred):
        print('> --------------- ')
        print(p.shape)
        print('> saving ---------- ')
        cv.imwrite('result/data_pred/cifar-{}-img.png'.format(str(i)), data[i % mod_size].numpy() * 255.)
        cv.imwrite('result/data_pred/cifar-{}-true.png'.format(str(i)), t * 255.)
        cv.imwrite('result/data_pred/cifar-{}-pred.png'.format(str(i)), p.numpy() * 255.)
        i += 1



# cv.imshow('y_pred', y_pred)
# cv.waitKey(0)
# cv.destroyAllWindows()
