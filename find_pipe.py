import json
import os

import numpy as np
import cv2
from test import post_processing


def find_pipe(image):
    gimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gimg = np.reshape(gimg, gimg.shape + (1,))

    circles = cv2.HoughCircles(gimg, cv2.HOUGH_GRADIENT, 2, 20, param1=1000, param2=50, minRadius=0, maxRadius=0)

    if circles is None:
        circles = []
    else:
        circles = np.uint16(np.around(circles))[0, :]

    for i in circles:
        return i


arr = dict()

for img in os.listdir('training_set3/y'):
    image = cv2.imread(os.path.join('training_set3', 'y', img))
    circle = find_pipe(image)
    if circle is not None:
        cir = [int(i) for i in list(circle)]
        arr.update({img.replace('in', 'out'): cir})

file = open('pipes.json', 'w')
json.dump(arr, file)
