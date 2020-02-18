import torch
from model import Model
import numpy as np
import os
import cv2
from config import size

def get_img_list(path):
    # files = glob(os.path.join(path, '*'))
    files = [os.path.join(path, '%d.jpg' % i) for i in range(856)]
    return files


def run_result():
    test_path = r'E:\Data\food_challenge2\test'
    result_path = 'result.csv'
    test_img_list = get_img_list(test_path)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model('BackBoneNet', device, learning_rate=1)
    model.load_model('model/BackBoneNet1.pth')
    writer = open(result_path, 'w')
    for i, filename in enumerate(test_img_list):
        image = cv2.imread(filename).astype(np.float32)
        pred_y = model.predict(image, (size, size))[0]
        writer.write('%s,%d\n' % (i, pred_y))
    writer.close()




if __name__ == '__main__':
    run_result()