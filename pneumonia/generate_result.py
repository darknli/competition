import torch
from pneumonia.pneunonia_model import Model
import numpy as np
import os
import cv2
from pneumonia.config import *

def get_img_list(path, num):
    # files = glob(os.path.join(path, '*'))
    files = [os.path.join(path, '%d.jpg' % i) for i in range(num)]
    return files


def run_result():
    test_path = r'E:\Data\pneumonia\test'
    result_path = 'result.csv'
    test_img_list = get_img_list(test_path, num_test)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model('inception_v3_google', num_classes, device, save_model_dir=model_path, learning_rate=learning_rate)
    model.load_model('model/inception_v3_google.pth')
    writer = open(result_path, 'w')

    model.model_eval()
    with torch.no_grad():
        for i, filename in enumerate(test_img_list):
            image = cv2.imread(filename).astype(np.float32)
            pred_y, prob = model.predict(image, (size, size), True)
            # writer.write('%s,%d,%.3f\n' % (i, pred_y[0], prob))
            writer.write('%s,%d\n' % (i, pred_y))
            print('已完成%s,%d' % (i, pred_y))
        writer.close()




if __name__ == '__main__':
    run_result()