import torch
from common.model_zoo.model import Model
import numpy as np
import os
import cv2
from food_challenge2.config import size, num_classes, num_test
from math import ceil

def get_img_list(path, num=856):
    # files = glob(os.path.join(path, '*'))
    files = [os.path.join(path, '%d.jpg' % i) for i in range(num)]
    return files


def run_result():
    test_path = r'E:\Data\food_challenge2\test'
    result_path = 'result_cmp.csv'
    test_img_list = get_img_list(test_path, num_test)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model('mobilenet_v2', num_classes, device, learning_rate=1)
    model.load_model('model/mobilenet_v22.pth')
    writer = open(result_path, 'w')

    model.model_eval()
    with torch.no_grad():
        for j, filename in enumerate(test_img_list):
            image = cv2.imread(filename).astype(np.float32)
            h, w = image.shape[:2]
            if h > w * 1.2:
                max_pred, max_prob = 0, 0
                slide = int(h*0.1)
                n = ceil((h - w) / slide)
                for i in range(n):
                    start = i * slide
                    end = min(h, start + w)
                    pred_y, prob = model.predict(image[start:end, :, :], (size, size), True)
                    if prob > max_prob:
                        max_pred = pred_y
                        max_prob = prob
            elif w > h * 1.2:
                max_pred, max_prob = 0, 0
                slide = int(h*0.1)
                n = ceil((w - h) / slide)
                for i in range(n):
                    start = i * slide
                    end = min(w, start + h)
                    pred_y, prob = model.predict(image[:, start:end, :], (size, size), True)
                    if prob > max_prob:
                        max_pred = pred_y
                        max_prob = prob
            else:
                max_pred, max_prob = model.predict(image, (size, size), True)
            writer.write('%s,%d,%.3f\n' % (j, max_pred, max_prob))
            # writer.write('%s,%d\n' % (j, max_pred))
            print('预测%d完成, 第%d类, 概率%.3f' % (j, max_pred, max_prob))
        writer.close()


# 目前26个错误
if __name__ == '__main__':
    run_result()
