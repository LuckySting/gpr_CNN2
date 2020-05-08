from random import Random
import numpy as np
import skimage.io as io
import skimage.transform as trans
import os


def format_image(image):
    image = image / 255
    image = trans.resize(image, (256, 256))
    image = np.reshape(image, image.shape + (1,))
    image = np.reshape(image, (1,) + image.shape)
    return image


def data_generator(path, start, length, flip=False, limit=None, rescale=True):
    i = 0
    random = Random()
    while True:
        ind = start + (i % length)
        img_x = io.imread(os.path.join(path, 'x', 'out_{}.png'.format(ind)), as_gray=True)
        img_y = io.imread(os.path.join(path, 'y', 'in_{}.png'.format(ind)), as_gray=True)
        if flip and random.randint(1, 2) == 1:
            img_x = np.flip(img_x, axis=1)
            img_y = np.flip(img_y, axis=1)
        if rescale:
            img_x = format_image(img_x)
            img_y = format_image(img_y)
        i += 1
        yield img_x, img_y
        if limit == i:
            break


def data_generator2(y_dict, path, start, length, flip=False, limit=None, rescale=True):
    i = 0
    random = Random()
    while True:
        ind = (start + i) % (start + length) + 1
        img_name = os.path.join(path, 'in', 'x', 'out_{}.png'.format(ind))
        img_x = io.imread(img_name, as_gray=True)
        y = np.array(y_dict['out_{}.png'.format(ind)]) / 256
        y = np.reshape(y, (1, 3))
        if flip and random.randint(1, 2) == 1:
            img_x = np.flip(img_x, axis=1)
        if rescale:
            img_x = format_image(img_x)
        i += 1
        yield img_x, y


def res_image(img):
    max_v = np.max(img)
    img /= max_v
    img *= 255
    return img


def save_result(result, start=1, path='.'):
    if not os.path.isdir(path):
        os.mkdir(path)

    for i, img in enumerate(result):
        io.imsave(os.path.join(path, 'predict_{}.png'.format(start + i)), res_image(img))

