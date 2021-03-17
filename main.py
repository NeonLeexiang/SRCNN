# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import cv2 as cv
import self_load_data
import numpy as np

# test = cv.imread('Train/t1.bmp')
# test_label = cv.resize(test, (128, 128), interpolation=cv.INTER_CUBIC)
# test_train = cv.resize(test, (32, 32), interpolation=cv.INTER_NEAREST)
# test_train = cv.resize(test_train, (128, 128), interpolation=cv.INTER_CUBIC)
#
# # img resize into
#
# cv.imshow('test_train', test_train)
# cv.imshow('test_label', test_label)
#
#
# cv.waitKey(0)
# cv.destroyAllWindows()


(train_images, train_labels), (test_images, test_labels) = self_load_data.load_data('cifar-10-python.tar')


def img_process_train(img):
    """
        resize it into 32*32 then resize it into 128*128 by using inter_cubic
        according to the paper we use bicubic methods to resize the img into the
        same size with High Resolution Image, then training the CNN model and output
        the Super Resolution image.
    :param img:
    :return:
    """
    train = cv.resize(img, (32, 32), interpolation=cv.INTER_NEAREST)
    train = cv.resize(train, (128, 128), interpolation=cv.INTER_CUBIC)
    return np.array(train).reshape((128, 128, 1))


def img_process_label(img):
    return np.array(cv.resize(img, (128, 128), interpolation=cv.INTER_CUBIC)).reshape((128, 128, 1)) / 255.

data = cv.cvtColor(test_images[0], cv.COLOR_RGB2GRAY)

test_label = cv.resize(data, (128, 128), interpolation=cv.INTER_CUBIC)
test_train = img_process_train(data)
print(test_train.shape)
print(test_train)
cv.imshow('true', data)
cv.imshow('test_train', test_train)
cv.imshow('test_label', test_label)


cv.waitKey(0)
cv.destroyAllWindows()
