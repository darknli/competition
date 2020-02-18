from data import get_data_generator
from model import Model
import torch
from generate_train_eval import generate_train_split
from config import *

def train():
    # generate_train_split(dataset_path, train_path, val_path, split_num)
    train_loader = get_data_generator(train_path, image_root, size, batch_size, True, num_workers, 'train')
    val_loader = get_data_generator(val_path, image_root, size, batch_size, False, num_workers, 'val')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model('inception_v3_google', num_classes, device, learning_rate=learning_rate)
    # model.load_model('model/BackBoneNet.pth')
    for epoch in range(epochs):
        print('第%d轮' % epoch)
        model.train(train_loader)
        model.eval(val_loader)


if __name__ == '__main__':
    train()