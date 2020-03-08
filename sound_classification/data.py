from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from common.utils.subprocess import *
from sound_classification.token2idx import convert2idx
import wavio
import numpy as np
from scipy import signal
import librosa
from tqdm import tqdm
from multiprocessing import Pool
from math import ceil
from time import time


class SoundDataset(Dataset):
    def __init__(self, filename, root_path, slide, mode):
        super(SoundDataset, self).__init__()
        self.anno_list = self.get_anno(filename, root_path)
        self.mode = mode
        self.len = len(self.anno_list)
        self.slide = slide
        self.pitch_shift = 0.3
        self.volumn_scale = 1.3
        self.time_stretch1, self.time_stretch2 = 0.05, 0.25

    def get_anno(self, files, root_path, workers=6):
        print('开始准备数据')
        start_time = time()
        anno_list = []

        task_pool = Pool(processes=workers)
        batch_size = ceil(len(files)/workers)

        all_results = []
        for i in range(workers):
            start = min(i * batch_size, len(files))
            end = min(start + batch_size, len(files))
            result = task_pool.apply_async(self._load_sound, (files[start: end], ))
            all_results.append(result)
        task_pool.close()
        task_pool.join()
        for result in all_results:
            anno_list.extend(result.get())

        # with tqdm(files) as pbar:
        #     for file in pbar:
        #         name = file.split('.')[0]
        #         fields = name.split('-')
        #         label1, label2 = fields[-2], fields[-1]
        #         label = int(label2)
        #         sig, fs = librosa.load(file)
        #         # data = self._mel_feature(file)
        #         # anno_list.append((np.expand_dims(data, axis=0), int(label)))
        #         anno_list.append((sig, fs, int(label)))

        finish_time = time()
        print('音频加载完成，花费了%ds' % (finish_time - start_time))
        return anno_list

    def _load_sound(self, files):
        anno_list = []
        for file in files:
            name = file.split('.')[0]
            fields = name.split('-')
            label1, label2 = fields[-2], fields[-1]
            label = int(label2)
            sig, fs = librosa.load(file)
            # data = self._mel_feature(file)
            # anno_list.append((np.expand_dims(data, axis=0), int(label)))
            anno_list.append((sig, fs, int(label)))
        return anno_list

    def __len__(self):
        return self.len

    def __getitem__(self, idx, istile=False):
        sig, fs, label = self.anno_list[idx]
        sig = sig.copy()
        if self.mode == 'train':
            # if np.random.random() < 0.8:
            #     sig = self._augment(sig, fs)
            start_idx = np.random.randint(0, self.slide)
        elif self.mode == 'val':
            start_idx = self.slide//2
        else:
            raise ValueError('mode有问题')
        data = librosa.feature.melspectrogram(y=sig, sr=fs)
        # data = librosa.feature.mfcc(y=sig, sr=fs)
        data = np.expand_dims(data, 0)
        if start_idx < self.slide:
            if istile:
                return np.tile(data[:, :, start_idx:-self.slide+start_idx], (3, 1, 1)), label
            else:
                return data[:, :, start_idx:-self.slide+start_idx], label
        else:
            if istile:
                return np.tile(data[:, :, start_idx:], (3, 1, 1)), label
            else:
                return data[:, :, start_idx:], label

    def _mel_feature(self, path):
        sig, fs = librosa.load(path)
        S = librosa.feature.melspectrogram(y=sig, sr=fs)
        return S

    def _augment(self, y, sr):
        rd1 = np.random.uniform(-self.pitch_shift, self.pitch_shift)
        volumn_up = np.random.uniform(max(1/self.volumn_scale, 0.6), self.volumn_scale)
        y = y*volumn_up
        # y = librosa.effects.pitch_shift(y, sr, n_steps=rd1)
        return y
        # if np.random.random() < 0.5:
        #     rd1 = np.random.uniform(self.pitch_shift1, self.pitch_shift2)
        #     y_ps = librosa.effects.pitch_shift(y, sr, n_steps=rd1)
        #     if y_ps != y:
        #         print('1', y_ps.shape, y.shape)
        #     return y_ps
        # else:
        #     ii = np.random.choice((-1, 1))
        #     rd2 = np.random.uniform(self.time_stretch1, self.time_stretch2)
        #     rd2 = 1.0 + ii * rd2
        #     y_ts = librosa.effects.time_stretch(y, rate=rd2)
        #     if y_ts != y:
        #         print('2', y_ts.shape, y.shape)
        #     return y_ts



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
        data = SoundDataset(filename, root_path, 30, mode)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader


