import numpy as np
import skimage.io as io
import skimage.transform as trans
import os

from keras.preprocessing.image import ImageDataGenerator


def format_image(image):
    img_x = image / 255
    img_x = trans.resize(img_x, (256, 256))
    img_x = np.reshape(img_x, img_x.shape + (1,))
    return np.reshape(img_x, (1,) + img_x.shape)


def aug_data_generator(path):
    x_gen = ImageDataGenerator(horizontal_flip=True)
    y_gen = ImageDataGenerator(horizontal_flip=True)

    x_images = x_gen.flow_from_directory(path + '/x', (256, 256), 'grayscale', seed=1)
    y_images = y_gen.flow_from_directory(path + '/y', (256, 256), 'grayscale', seed=1)

    train_generator = zip(x_images, y_images)

    for (img_x, img_y) in train_generator:
        yield format_image(img_x), format_image(img_y)


def data_generator(path, start, length):
    i = 0
    while True:
        ind = start + (i % length)
        img_x = format_image(io.imread(os.path.join(path, 'x', 'out_{}.png'.format(ind)), as_gray=True))
        img_y = format_image(io.imread(os.path.join(path, 'y', 'in_{}.png'.format(ind)), as_gray=True))
        i += 1
        yield img_x, img_y


def save_result(path, result, start=1):
    if not os.path.isdir(path):
        os.mkdir(path)
    for i, img in enumerate(result):
        max_v = np.max(img)
        img /= max_v
        img *= 255
        io.imsave(os.path.join(path, "predict_{}.png".format(start + i)), img)
