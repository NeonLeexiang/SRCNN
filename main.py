# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import cv2 as cv

test = cv.imread('Train/t1.bmp')
test_label = cv.resize(test, (128, 128), interpolation=cv.INTER_CUBIC)
test_train = cv.resize(test, (32, 32), interpolation=cv.INTER_NEAREST)
test_train = cv.resize(test_train, (128, 128), interpolation=cv.INTER_CUBIC)

# img resize into

cv.imshow('test_train', test_train)
cv.imshow('test_label', test_label)


cv.waitKey(0)
cv.destroyAllWindows()
