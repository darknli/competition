import torch
from sound_classification.sound_model import Model
import numpy as np
import os
import cv2
from sound_classification.config import num_classes, num_test
import librosa

def get_img_list(path, num=856):
    # files = glob(os.path.join(path, '*'))
    files = [os.path.join(path, '%d.wav' % i) for i in range(num)]
    return files


def run_result():
    test_path = r'D:\temp_data\sound_classification_50\test'
    result_path = 'result.csv'
    test_img_list = get_img_list(test_path, num_test)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model('mobilenet_v2', num_classes, device, learning_rate=1)
    model.load_model('model/0_ohem_mobilenet_v2.pth')
    writer = open(result_path, 'w')

    model.model_eval()
    with torch.no_grad():
        for j, filename in enumerate(test_img_list):
            sig, fs = librosa.load(filename)
            S = librosa.feature.melspectrogram(y=sig, sr=fs)
            max_pred, max_prob = model.predict(np.tile(np.expand_dims(S, 0)[:, :, 15:-15], (3, 1, 1)), True)
            # writer.write('%s,%d,%.3f\n' % (j, max_pred, max_prob))
            writer.write('%s,%d\n' % (j, max_pred))
            print('预测%d完成, 第%d类, 概率%.3f' % (j, max_pred, max_prob))
        writer.close()


# 目前26个错误
if __name__ == '__main__':
    run_result()
