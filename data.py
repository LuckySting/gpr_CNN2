import numpy as np
import skimage.io as io
import skimage.transform as trans
import os


def data_generator(path, start, length):
    i = start
    while True:
        img_x = io.imread(os.path.join(path, 'x', 'out_{}.png'.format(i % length + 1)), as_gray=True)
        img_x = img_x / 255
        img_x = trans.resize(img_x, (256, 256))
        img_x = np.reshape(img_x, img_x.shape + (1,))
        img_x = np.reshape(img_x, (1,) + img_x.shape)
        img_y = io.imread(os.path.join(path, 'y', 'in_{}.png'.format(i % length + 1)), as_gray=True)
        img_y = img_y / 255
        img_y = trans.resize(img_y, (256, 256))
        img_y = np.reshape(img_y, img_y.shape + (1,))
        img_y = np.reshape(img_y, (1,) + img_y.shape)
        i += 1
        yield img_x, img_y


def save_result(path, result):
    for i, img in enumerate(result):
        max_v = np.max(img)
        img /= max_v
        img *= 255
        io.imsave(os.path.join(path, "%d_predict.png" % i), img)
