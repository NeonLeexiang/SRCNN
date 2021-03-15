"""
    date:       2021/3/9 4:04 下午
    written by: neonleexiang
"""
from model import SRCNN
from data_preprocessing import SRCNNLoader
import tensorflow as tf
import cv2 as cv
import numpy as np

# parameters
num_epochs = 5
batch_size = 30
learning_rate = 0.001

model = SRCNN()
data_loader = SRCNNLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    # print(X.shape)
    # print(model(X))
    # print(y)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.mean_squared_error(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print('training ended')

i = 0
for img in data_loader.test_data:
    i += 1
    img = np.expand_dims(img, axis=0)
    y_pred = model.predict(img)
    print('> --------------- ')
    print(y_pred.shape)
    print('> saving ---------- ')
    # print(img[0])
    # print(y_pred[0])
    # break
    cv.imwrite('result/data_pred/{}-img.png'.format(str(i)), img[0] * 255)
    cv.imwrite('result/data_pred/{}-pred.png'.format(str(i)), y_pred[0] * 255)


# sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
# num_batches = int(data_loader.num_test_data // batch_size)
# for batch_index in range(num_batches):
#     start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
#     y_pred = model.predict(data_loader.test_data[start_index: end_index])
#     sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
# print("test accuracy: %f" % sparse_categorical_accuracy.result())

