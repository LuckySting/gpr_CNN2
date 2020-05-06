from data import data_generator, save_result, res_image
from unet import unet
import cv2
from post_processing import post_processing
import numpy as np
import skimage.transform as trans

start = 50
length = 20
test_generator = data_generator('training_set', start, length)
model = unet()
weight_name = 'second_generation_1'
model.load_weights('{}.hdf5'.format(weight_name))
result = model.predict_generator(test_generator, steps=length, verbose=1)
for i, img in enumerate(result):
    img_n = res_image(img).astype('uint8')
    img_n = cv2.cvtColor(np.asarray(img_n), cv2.COLOR_GRAY2BGR)
    img_n = post_processing(img_n)
    io = [cv2.imread('training_set/x/out_{}.png'.format(start+i)), cv2.imread('training_set/y/in_{}.png'.format(start+i))]
    cv2.imshow('input'.format(i), trans.resize(io[0], (256, 256)))
    cv2.imshow('predict', img_n)
    cv2.imshow('reality', trans.resize(io[1], (256, 256)))
    cv2.waitKey()

# save_result(result, start=215, path='./{}'.format(weight_name))

