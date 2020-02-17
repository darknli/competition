from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from utils.subprocess import *


class FoodDataset(Dataset):
    def __init__(self, filename, root_path, size, mode):
        super(FoodDataset, self).__init__()
        self.anno_list = self.get_anno(filename, root_path)
        self.transforms = self.get_transforms(mode)
        self.mode = mode
        self.len = len(self.anno_list)
        self.size = (size, size)

    def get_transforms(self, mode):
        from torchvision import transforms
        need_trans = [transforms.ToTensor()]
        if mode == 'train':
            need_trans.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
        transform = transforms.Compose(need_trans)
        return transform

    def get_anno(self, filename, root_path):
        anno_list = []
        with open(filename) as f:
            for line in f:
                idx, label = line.strip().split(',')
                anno_list.append(('%s/%s.jpg' % (root_path, idx), int(label)))
        return anno_list

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        path, label = self.anno_list[idx]
        image = cv2.imread(path).astype(np.float32)
        image = square(image, self.size)
        if self.mode == 'train' and np.random.random() < 0.5:
            image = self.augment(image, bright=10, contrast=0.1, guassion_blur=False)
        image = subprocess(image)
        # image = self.transforms(image)
        return image, label

    def augment(self, image, flip=True, bright=0, contrast=None, guassion_blur=False):
        if flip:
            image = image[:, ::-1, :]
        if bright:
            bright = np.random.randint(-bright, bright)
            image += bright
            image = np.clip(image, 0, 255)
        if contrast:
            contrast = np.random.uniform(1-contrast, 1+contrast)
            image *= contrast
            image = np.clip(image, 0, 255).astype(np.uint8).astype(np.float32)
        if guassion_blur and np.random.random() < 0.5:
            blur_level = np.random.randint(1, 3)*2 + 1
            image = cv2.GaussianBlur(image, (blur_level, blur_level), np.sqrt(blur_level/6))
        return image


def get_data_generator(filename, root_path, size, batch_size, shuffle, num_workers, mode):
    data = FoodDataset(filename, root_path, size, mode)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
