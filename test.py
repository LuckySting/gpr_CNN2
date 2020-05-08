import numpy as np
import cv2


def nothing(_):
    pass


def post_processing(image):
    cv2.namedWindow('thershold')
    cv2.createTrackbar('thershold1', 'thershold', 1, 2000, nothing)
    cv2.createTrackbar('thershold2', 'thershold', 1, 2000, nothing)
    cv2.setTrackbarPos('thershold1', 'thershold', 1000)
    cv2.setTrackbarPos('thershold2', 'thershold', 50)

    while True:
        imgC = cv2.medianBlur(image, 11)
        _, imgT = cv2.threshold(imgC, 100, 255, cv2.THRESH_BINARY)
        imgTE = cv2.Canny(imgT, 150, 800, apertureSize=5)
        p1 = cv2.getTrackbarPos('thershold1', 'thershold')
        p2 = cv2.getTrackbarPos('thershold2', 'thershold')
        print('{} {}'.format(p1, p2))
        circles = cv2.HoughCircles(imgTE, cv2.HOUGH_GRADIENT, dp=3, minDist=50, param1=p1, param2=p2, minRadius=5,
                                   maxRadius=100)
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
            if mean_circle_color > 70:
                continue
            cv2.circle(imgC, (x, y), r + 2, (0, 255, 0), 1)

        cv2.imshow('canny', imgTE)
        cv2.imshow('result', imgC)
        cv2.waitKey()
        if cv2.getWindowProperty('result', 0) == -1:
            break

    return imgC
