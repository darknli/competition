import cv2
import numpy as np


def square(image, size):
    h, w = image.shape[:2]
    if h > w:
        h_s = (h - w) // 2
        image = image[h_s:h_s + w, :, :]
    elif h < w:
        w_s = (w - h) // 2
        image = image[:, w_s: w_s + h, :]
    return cv2.resize(image, size)


def subprocess(image):
    image = np.transpose(image, (2, 1, 0))
    image -= 127
    image /= 255
    return image