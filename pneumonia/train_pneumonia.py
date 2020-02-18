from pneumonia.data import get_data_generator
from common.model_zoo.model import Model
import torch
from common.utils.generate_train_eval import generate_train_split
from pneumonia.config import *



def train():
    # generate_train_split(dataset_path, train_path, val_path, split_num)
    train_loader = get_data_generator(train_path, image_root, size, batch_size, True, num_workers, 'train', True)
    val_loader = get_data_generator(val_path, image_root, size, batch_size, False, num_workers, 'val', True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model('mobilenet_v2', num_classes, device, save_model_dir=model_path, learning_rate=learning_rate)
    # model.set_num_fintune_layers(10)
    # model.load_model('model/BackBoneNet.pth')
    for epoch in range(epochs):
        print('第%d轮' % epoch)
        model.train(train_loader)
        model.eval(val_loader)


if __name__ == '__main__':
    train()