import torch
from pneumonia.pneunonia_model import Model, RegBoxModel
import numpy as np
import os
import cv2
from pneumonia.config import *

def get_img_list(path, num):
    # files = glob(os.path.join(path, '*'))
    files = [os.path.join(path, '%d.jpg' % i) for i in range(num)]
    return files


def run_result():
    test_path = r'D:\temp_data\pneumonia\test'
    result_path = 'result.csv'
    test_img_list = get_img_list(test_path, num_test)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    incep_model = Model('inception_v3_google', num_classes, device, save_model_dir=model_path, learning_rate=learning_rate)
    reg_model = RegBoxModel('inception_v3_google', 128, device, save_model_dir=model_path, learning_rate=learning_rate,
                prefix='Reg_')
    mobile_model = Model('mobilenet_v2', num_classes, device, save_model_dir=model_path,
                        learning_rate=learning_rate)

    incep_model.load_model('model/inception_v3_google.pth')
    reg_model.load_model('model/Reg_inception_v3_google.pth')
    mobile_model.load_model('model/mobilenet_v2.pth')

    writer = open(result_path, 'w')

    incep_model.model_eval()
    reg_model.model_eval()
    mobile_model.model_eval()
    with torch.no_grad():
        for i, filename in enumerate(test_img_list):
            image = cv2.imread(filename).astype(np.float32)
            pred_y_reg = reg_model.predict(image, (size, size))
            pred_y_incep = incep_model.predict(image, (size, size), False)
            pred_y_mob = mobile_model.predict(image, (size, size), False)
            # pred_y, prob = model.predict(image, (size, size), True)
            # writer.write('%s,%d,%.3f\n' % (i, pred_y[0], prob))
            result = np.zeros((num_classes))
            print(pred_y_reg, pred_y_incep, pred_y_mob)
            result[pred_y_reg[0]] += 1
            result[pred_y_incep[0]] += 1
            result[pred_y_mob[0]] += 1
            idx = np.argmax(result)
            writer.write('%s,%d\n' % (i, idx))
            print('已完成%s,%d' % (i, idx))
        writer.close()




if __name__ == '__main__':
    run_result()