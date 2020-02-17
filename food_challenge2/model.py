import torch
from torch import nn
import torchsummary
from tqdm import tqdm
from utils.metric import get_acc
from utils.loss import cross_entroy
from time import time
from numpy import inf
import os

class BackBoneNet(nn.Module):
    def __init__(self, base_num=4):
        super(BackBoneNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4*base_num, kernel_size=3)

        self.conv2 = nn.Conv2d(in_channels=4*base_num, out_channels=8*base_num, kernel_size=3)

        self.conv3 = nn.Conv2d(in_channels=8*base_num, out_channels=16*base_num, kernel_size=3)

        self.conv4 = nn.Conv2d(in_channels=16*base_num, out_channels=32*base_num, kernel_size=3)

        self.fc1 = nn.Linear(in_features=32*base_num, out_features=16*base_num)
        self.fc2 = nn.Linear(in_features=16*base_num, out_features=4)

        self.pool = nn.MaxPool2d(3, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.pool(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.pool(y)
        y = self.conv3(y)
        y = self.relu(y)
        y = self.pool(y)
        y = self.conv4(y)
        y = self.relu(y)
        y = torch.max(y, dim=-1)[0]
        y = torch.max(y, dim=-1)[0]
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return y

class Model:
    def __init__(self, model_type, device, save_model_dir='model', opt_mode='adam'):
        if model_type == 'BackBoneNet':
            self.model = BackBoneNet(4).to(device)
        else:
            raise ValueError('沒有這個模型！')
        self.device = device
        self.optimizer = self.get_optimizer(opt_mode)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=8,
            verbose=True, threshold_mode='abs', min_lr=10e-7
        )
        self.min_loss = inf
        self.save_model_path = os.path.join(save_model_dir, model_type)

    def get_optimizer(self, mode='adam', learning_rate=10e-2):
        if mode == 'adam':
            from torch.optim import Adam
            return Adam(self.model.parameters(), lr=learning_rate)
        elif mode == 'sgd':
            from torch.optim import SGD
            return SGD(self.model.parameters(), lr=learning_rate)


    def train(self, data_loader):
        start = time()
        self.model.train()

        total_loss = 0
        total_count = 0
        with tqdm(data_loader) as pbar:
            for image, label in pbar:
                self.optimizer.zero_grad()
                image = image.to(self.device)
                label = label.to(self.device)
                pred_y = self.model(image)
                acc = get_acc(pred_y, label).item()
                loss, k = cross_entroy(pred_y, label, 0.8)
                loss.backward()
                self.optimizer.step()
                loss += loss.item()*k
                total_count += k
                pbar.set_postfix(loss=loss.item(), acc=acc)

        end = time()
        print('训练耗时:%ds, loss=%.3f' % (end - start, total_loss/total_count))

    def eval(self, data_loader, save_best_model=True):
        start = time()
        self.model.train()

        total_loss = 0
        total_count = 0
        total_pred_result = []
        total_true_result = []
        with torch.no_grad():
            for image, label in data_loader:
                self.optimizer.zero_grad()
                image = image.to(self.device)
                label = label.to(self.device)
                pred_y = self.model(image)
                loss, _ = cross_entroy(pred_y, label, 1)
                loss += loss.item()
                total_count += 1
                total_pred_result.append(pred_y)
                total_true_result.append(label)
        pred_y = torch.cat(total_pred_result, dim=0)
        true_y = torch.cat(total_true_result, dim=0)
        acc = get_acc(pred_y, true_y).item()
        end = time()
        total_loss /= total_count
        self.scheduler.step(total_loss)
        if save_best_model:
            self.save_model(total_loss, acc)
        print('验证耗时:%ds, loss=%.3f, acc=%.3f' % (end - start, total_loss, acc))

    def save_model(self, loss, acc):
        if loss < self.min_loss:
            self.min_loss = loss
            params = {}
            params['loss'] = loss
            params['acc'] = acc
            params['state_dict'] = self.model.state_dict()
            torch.save(params, self.save_model_path)
            print('min_loss update to %d' % loss)

if __name__ == '__main__':
    net = BackBoneNet()
    net.eval()
    torchsummary.summary(net, (3, 256, 256))