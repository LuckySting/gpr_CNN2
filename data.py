import numpy as np
import skimage.io as io
import skimage.transform as trans
import os

from keras.preprocessing.image import ImageDataGenerator


def format_image(image):
    img = image / 255
    img = trans.resize(img, (256, 256))
    img = np.reshape(img, img.shape + (1,))
    return np.reshape(img, (1,) + img.shape)


def normalize_image(image):
    img = image[0, :, :, 0]
    img /= 255
    img = np.reshape(img, img.shape + (1,))
    return np.reshape(img, (1,) + img.shape)


def aug_data_generator(path):
    x_gen = ImageDataGenerator(horizontal_flip=True)
    y_gen = ImageDataGenerator(horizontal_flip=True)

    x_images = x_gen.flow_from_directory(path, (256, 256), 'grayscale', classes=['x'], batch_size=1, seed=1)
    y_images = y_gen.flow_from_directory(path, (256, 256), 'grayscale', classes=['y'], batch_size=1, seed=1)

    train_generator = zip(x_images, y_images)

    for (img_x, img_y) in train_generator:
        yield normalize_image(img_x[0]), normalize_image(img_y[0])


def data_generator(path, start, length):
    i = 0
    while True:
        ind = start + (i % length)
        img_x = format_image(io.imread(os.path.join(path, 'x', 'out_{}.png'.format(ind)), as_gray=True))
        img_y = format_image(io.imread(os.path.join(path, 'y', 'in_{}.png'.format(ind)), as_gray=True))
        i += 1
        yield format_image(img_x[0]), format_image(img_y[0])


def save_result(path, result, start=1):
    if not os.path.isdir(path):
        os.mkdir(path)
    for i, img in enumerate(result):
        max_v = np.max(img)
        img /= max_v
        img *= 255
        io.imsave(os.path.join(path, "predict_{}.png".format(start + i)), img)
