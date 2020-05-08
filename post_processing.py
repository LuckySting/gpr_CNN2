import numpy as np
import cv2


def post_processing(image):
    imgC = cv2.medianBlur(image, 11)
    _, imgT = cv2.threshold(imgC, 150, 255, cv2.THRESH_BINARY)
    imgTE = cv2.Canny(imgT, 150, 800, apertureSize=5)
    cv2.imshow('canny', imgTE)

    circles = cv2.HoughCircles(imgTE, cv2.HOUGH_GRADIENT, dp=2, minDist=65, param1=1000, param2=50, minRadius=5,
                               maxRadius=50)
    if circles is None:
        circles = []
    else:
        circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        if r > x:
            r = x - 1
        if r > y:
            r = y - 1
        yy, xx = np.ogrid[-r: r, -r: r]
        index = xx ** 2 + yy ** 2 <= r ** 2
        index = np.reshape(index, index.shape + (1,))
        index = np.repeat(index, 3, axis=2)
        try:
            mean_circle_color = np.mean(imgC[y - r:y + r, x - r:x + r, :][index])
        except:
            continue
        if mean_circle_color > 150:
            continue
        cv2.circle(imgC, (x, y), r + 2, (0, 0, 0), cv2.FILLED)

    return imgC
