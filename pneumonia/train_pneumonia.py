from pneumonia.data import get_data_generator
from pneumonia.pneunonia_model import BoxModel, RegBoxModel, Model
import torch
from common.utils.generate_train_eval import generate_train_split
from pneumonia.split_data import split_data
from pneumonia.config import *



def train():
    generate_train_split(dataset_path, train_path, val_path, split_num)
    split_data(dataset_path, box_path, train_path, val_path, split_num)
    train_loader = get_data_generator(train_path, image_root, size, batch_size, True, num_workers, 'train', False)
    val_loader = get_data_generator(val_path, image_root, size, batch_size, False, num_workers, 'val', False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = BoxModel('inception_v3_google', num_classes, 128, device, save_model_dir=model_path,
    # learning_rate=learning_rate, prefi='Cls_')
    # model = RegBoxModel('inception_v3_google', 128, device, save_model_dir=model_path, learning_rate=learning_rate,
    #                     prefix='Reg_')
    model = Model('mobilenet_v2', num_classes, device, save_model_dir=model_path,
                  learning_rate=learning_rate, prefix='')
    # model.set_num_fintune_layers(10)
    # model.load_model('model/inception_v3_google.pth')
    for epoch in range(epochs):
        print('第%d轮' % epoch)
        model.train(train_loader)
        model.eval(val_loader)


if __name__ == '__main__':
    train()