import os, sys, glob, argparse
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from car_classification.data import testDataset
from car_classification.car_model import Model

if __name__ == '__main__':

    test_jpg = [r'D:\temp_data\car\test/{0}.jpg'.format(x) for x in range(0, 3450)]
    test_jpg = np.array(test_jpg)

    test_pred = None
    for model_path in ['model_%d.pth' % i for i in range(5)]:
        test_loader = torch.utils.data.DataLoader(
            testDataset(test_jpg,
                      transforms.Compose([
                          transforms.Resize([256, 256]),
                          # transforms.Scale(299),
                          # transforms.RandomResizedCrop(224),
                          # transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])
                      ), batch_size=64, shuffle=False, num_workers=6, pin_memory=True
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Model("wide_resnet50_2", None, device=device)
        model.model.load_state_dict(torch.load(model_path))
        if test_pred is None:
            test_pred = model.predict(test_loader, 1)
        else:
            test_pred += model.predict(test_loader, 1)
    test_csv = pd.DataFrame()
    test_csv[0] = list(range(0, 3450))
    test_csv[1] = np.argmax(test_pred, 1)
    test_csv.to_csv('tmp5.csv', index=None, header=None)
    #test_pred.astype(int).iloc[:].to_csv('tmp.csv', index=None, header=None)
