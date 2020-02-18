import cv2
import numpy as np


def square(image, size, mode='val'):
    h, w = image.shape[:2]
    if h > w:
        h_s = (h - w) // 2
        image = image[h_s:h_s + w, :, :]
    elif h < w:
        w_s = (w - h) // 2
        image = image[:, w_s: w_s + h, :]
    if mode == 'train' and np.random.random() < 0.8:
        delta_side = np.random.randint(1, int(image.shape[0]*0.03))
        image = image[delta_side:-delta_side, delta_side:-delta_side, :]
    return cv2.resize(image, size)


def subprocess(image):
    image = np.transpose(image, (2, 1, 0))
    image -= 127
    image /= 255
    return image