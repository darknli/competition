from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from common.utils.subprocess import *


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
                fields = line.strip().split(',')
                idx, label, box = fields[0], fields[1], fields[2:]
                anno_list.append(('%s/%s.jpg' % (root_path, idx), int(label), [float(n) for n in box]))
        return anno_list

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        path, label, box = self.anno_list[idx]
        image = cv2.imread(path).astype(np.float32)
        w, h = image.shape[:2]
        image = cv2.resize(image, None, fx=0.25, fy=0.25)
        # image = square(image, self.size, self.mode)
        if self.mode == 'train' and np.random.random() < 0.5:
            image = self.augment(image, bright=10, contrast=0.1, rotate=True, guassion_blur=False)
        image = subprocess(image)
        # image = self.transforms(image)
        box[0], box[1], box[2], box[3] = box[0]/w, box[1]/w, box[2]/h, box[3]/h
        return image, label, np.array(box, dtype=np.float32)

    def augment(self, image, flip=True, bright=0, contrast=None, rotate=False, guassion_blur=False):
        if flip and np.random.random() < 0.5:
            image = image[:, ::-1, :]
        if flip and np.random.random() < 0.5:
            image = image[::-1, :, :]
        if rotate:
            du = np.random.choice([90, -90, 180])
            h, w = image.shape[:2]
            center = (w//2, h//2)
            m = cv2.getRotationMatrix2D(center, du, 1.0)
            image = cv2.warpAffine(image, m, (w, h))
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


class BalanceFoodDataset(FoodDataset):
    def __init__(self, **kwargs):
        super(BalanceFoodDataset, self).__init__(**kwargs)

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
        data = BalanceFoodDataset(filename=filename, root_path=root_path, size=size, mode=mode)
    else:
        data = FoodDataset(filename, root_path, size, mode)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


