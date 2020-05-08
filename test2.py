import itertools
import cv2


def find_params(image):
    img = image
    for dp, param2, param1 in itertools.product(range(1, 5), range(1, 200), range(1, 2000)):
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=dp, minDist=50, param1=param1, param2=param2, minRadius=5,
                                   maxRadius=100)
        if circles != None and len(circles) == 1:

            print('{} {} {}'.format(dp, param1, param2))
