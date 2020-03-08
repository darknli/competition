from sound_classification.config import *
from glob import glob
from sound_classification.data import get_data_generator
import os
import numpy as np
import torch
from sound_classification.sound_model import Model

def train():
    files = glob(os.path.join(image_root, '*'))
    np.random.shuffle(files)
    num_train = int(split_num * len(files))
    train = files[num_train:]
    val = files[:num_train]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model('mobilenet_v2', num_classes, device, save_model_dir=model_path,
        learning_rate=learning_rate, prefix='mfcc_')
    train_loader = get_data_generator(train, image_root, size, batch_size, True, num_workers, 'train', False)
    # for x, y in train_loader:
    #     print(x.shape, y.shape)
    val_loader = get_data_generator(val, image_root, size, batch_size, False, num_workers, 'val', False)
    # model.load_model('model/1_ohemmobilenet_v2.pth')
    for epoch in range(epochs):
        print('第%d轮' % epoch)
        model.train(train_loader)
        model.eval(val_loader)


if __name__ == '__main__':
    train()
    # import torchsummary
    #
    # model = Model('squeezenet1_0', num_classes, 'cuda', save_model_dir=model_path,
    #       learning_rate=learning_rate, prefix='').model
    # torchsummary.summary(model, (3, 256, 256))