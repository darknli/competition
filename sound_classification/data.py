from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from common.utils.subprocess import *
from sound_classification.token2idx import convert2idx
import wavio
import numpy as np
from scipy import signal
import librosa


class SoundDataset(Dataset):
    def __init__(self, filename, root_path, size, mode):
        super(SoundDataset, self).__init__()
        self.anno_list = self.get_anno(filename, root_path)
        self.mode = mode
        self.len = len(self.anno_list)
        self.size = (size, size)
        self.slide = 30

    def get_anno(self, files, root_path):
        anno_list = []
        for file in files:
            name = file.split('.')[0]
            fields = name.split('-')
            label1, label2 = fields[-2], fields[-1]
            # label = convert2idx(label1, label2)
            # print(label1, label2, label)
            label = int(label2)
            data = self._mel_feature(file)
            anno_list.append((np.expand_dims(data, axis=0), int(label)))
        return anno_list

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data, label = self.anno_list[idx]
        if self.mode == 'train':
            start_idx = np.random.randint(0, self.slide)
        elif self.mode == 'val':
            start_idx = self.slide//2
        else:
            raise ValueError('mode有问题')
        if start_idx < self.slide:
            return np.tile(data[:, :, start_idx:-self.slide+start_idx], (3, 1, 1)), label
        else:
            return np.tile(data[:, :, start_idx:], (3, 1, 1)), label

    def _mel_feature(self, path):
        sig, fs = librosa.load(path)
        S = librosa.feature.melspectrogram(y=sig, sr=fs)
        return S

    # def



class BalanceDataset(SoundDataset):
    def __init__(self, **kwargs):
        super(BalanceDataset, self).__init__(**kwargs)

    def get_anno(self, filename, root_path):
        anno_dict = {}
        with open(filename) as f:
            for line in f:
                idx, label = line.strip().split(',')
                if label not in anno_dict:
                    anno_dict[label] = []
                anno_dict[label].append('%s/%s.jpg' % (root_path, idx))
        self.num_classes = len(anno_dict)
        anno_list = [anno_dict[str(i)] for i in range(self.num_classes)]
        return anno_list

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.num_classes)
        path = np.random.choice(self.anno_list[idx])
        image = cv2.imread(path).astype(np.float32)
        image = square(image, self.size, self.mode)
        if self.mode == 'train' and np.random.random() < 0.5:
            image = self.augment(image, bright=10, contrast=0.1, rotate=True, guassion_blur=False)
        image = subprocess(image)
        return image, idx



def get_data_generator(filename, root_path, size, batch_size, shuffle, num_workers, mode, is_balance=False):
    if is_balance:
        data = BalanceDataset(filename=filename, root_path=root_path, size=size, mode=mode)
    else:
        data = SoundDataset(filename, root_path, size, mode)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


