import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL

class QRDataset(Dataset):
    def __init__(self, df, transform=None, cut_ratio=0.2):
        self.df = df
        self.cut_ratio = cut_ratio
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.df[0].iloc[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        #print(np.array(self.df.iloc[index, 1]))
        return img, torch.from_numpy(np.array(self.df.iloc[index, 1]))

    def __len__(self):
        return len(self.df)


class pDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(np.array(int('PNEUMONIA' in self.img_path[index])))

    def __len__(self):
        return len(self.img_path)