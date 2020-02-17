from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2


class FoodDataset(Dataset):
    def __init__(self, filename, root_path, mode):
        super(FoodDataset, self).__init__()
        self.anno_list = self.get_anno(filename, root_path)
        self.transforms = self.get_transforms(mode)
        self.len = len(self.anno_list)

    def get_transforms(self, mode):
        from torchvision import transforms
        need_trans = [ transforms.ToPILImage(), transforms.ToTensor()]
        if mode == 'train':
            need_trans.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
        transform = transforms.Compose(need_trans)
        return transform

    def get_anno(self, filename, root_path):
        anno_list = []
        with open(filename) as f:
            for line in f:
                idx, label = line.split()
                anno_list.append(('%s/%d.jpg' % (root_path, idx), label))
        return anno_list

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        path, label = self.anno_list[idx]
        image = cv2.imread(path)
        image = self.transforms(image)
        return image, label

def get_data_generator(filename, root_path, batch_size, shuffle, num_workers, mode):
    data = FoodDataset(filename, root_path, mode)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader