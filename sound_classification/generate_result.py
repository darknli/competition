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
    result_path = '1result.csv'
    test_img_list = get_img_list(test_path, num_test)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_list = []
    model_path_list = [
        # '2_mobilenet_v2.pth',
        # '3_mobilenet_v2.pth',
        # '4_mobilenet_v2.pth',
        'mobilenet_v2.pth'
    ]
    for load_path in model_path_list:
        model = Model('mobilenet_v2', num_classes, device, learning_rate=1)
        model.load_model('model/%s' % load_path)
        model.model_eval()
        model_list.append(model)

    writer = open(result_path, 'w')

    with torch.no_grad():
        for j, filename in enumerate(test_img_list):
            sig, fs = librosa.load(filename)
            # S = librosa.feature.melspectrogram(y=sig, sr=fs)
            S = librosa.feature.mfcc(y=sig, sr=fs)
            data = np.tile(np.expand_dims(S, 0)[:, :, 15:-15], (3, 1, 1))
            max_prob = 0
            max_pred = 0
            for model in model_list:
                pred, prob = model.predict(data, True)
                if prob > max_prob:
                    max_prob = prob
                    max_pred = pred
            # y = np.zeros((num_classes))
            # for model in model_list:
            #     pred = model.predict(data, False)
            #     y[pred] += 1
            # max_pred = np.argmax(y)
            # writer.write('%s,%d,%.3f\n' % (j, max_pred, max_prob))
            writer.write('%s,%d\n' % (j, max_pred))
            print('预测%d完成, 第%d类, 概率%.3f' % (j, max_pred, max_prob))
        writer.close()


# 目前26个错误
if __name__ == '__main__':
    run_result()
