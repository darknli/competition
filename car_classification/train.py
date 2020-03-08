import os, sys, glob
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from torchvision import transforms
import torch.utils.data
from car_classification.data import trainDataset
from car_classification.loss import *
from car_classification.car_model import Model
from car_classification.config import *
import numpy as np

def get_anno(path, images_path):
    data = []
    with open(path) as f:
        for line in f:
            idx, label = line.strip().split(',')
            data.append((os.path.join(images_path, idx), int(label)))
    return np.array(data)

if __name__ == '__main__':

    data = get_anno(train_anno_path, train_data_path)

    skf = KFold(n_splits=5, random_state=233, shuffle=True)
    for flod_idx, (train_indices, val_indices) in enumerate(skf.split(data)):
        train_loader = torch.utils.data.DataLoader(
            trainDataset(data[train_indices],
                      transforms.Compose([
                          transforms.Resize([299, 299]),
                          transforms.RandomRotation(15),
                          transforms.RandomChoice([transforms.Resize([256, 256]), transforms.CenterCrop([256, 256])]),
                          # transforms.RandomResizedCrop(224),
                          # transforms.Resize([ 256, 256]),
                          # transforms.CenterCrop(224),
                          # transforms.RandomChoice([transforms.RandomHorizontalFlip(),medianBlur]),
                          transforms.ColorJitter(brightness=0.2, contrast=0.2),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ]
                      )
                      ),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            trainDataset(data[val_indices],
                      transforms.Compose([
                          transforms.Resize([256, 256]),
                          # transforms.Scale(299),
                          # transforms.RandomResizedCrop(224),
                          # transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])
                      ), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        criterion = FocalLoss(0.5)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Model(model_type, num_classes, criterion, device=device, suffix=str(flod_idx))
        for epoch in range(epochs):
            print('Epoch: ', epoch)

            model.fit(train_loader)
            model.validate(val_loader)
