import torch
from torch import nn
import torchsummary
from torchvision import models
from tqdm import tqdm
from utils.metric import get_acc
from utils.loss import cross_entroy
from time import time
from numpy import inf
import os
from utils.subprocess import *
import torch

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

class PretrainModels(nn.Module):
    def __init__(self, model_type, num_classes):
        super(PretrainModels, self).__init__()
        if model_type == 'inception_v3_google':
            self.model = models.inception_v3(pretrained=True, aux_logits=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_type == 'resnet50':
            self.model = models.resnet50(pretrained=True,)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_type == 'resnet152':
            self.model = models.resnet152(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_type == 'resnext101_32x8d':
            self.model = models.resnext101_32x8d(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_type == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features,
                                                            num_classes)
        elif model_type == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features,
                                                            num_classes)
        elif model_type == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=True)
            self.model.classifier._modules['1'] = nn.Linear(self.model.classifier._modules['1'].in_features,
                                                            num_classes)
        else:
            raise ValueError('PretrainModels沒有這個模型！')
        self.model_type = model_type

    def forward(self, x):
        y = self.model(x)
        return y

class Model:
    def __init__(self, model_type, num_classes, device, save_model_dir='model', opt_mode='adam', learning_rate=10e-2):
        if model_type == 'BackBoneNet':
            self.model = BackBoneNet(4)
        else:
            self.model = PretrainModels(model_type, num_classes)

        self.model = self.model.to(device)
        self.device = device
        self.optimizer = self.get_optimizer(opt_mode, learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=8,
            verbose=True, threshold_mode='abs', min_lr=10e-7
        )
        self.min_loss = inf
        self.save_model_dir = save_model_dir
        self.save_model_path = os.path.join(save_model_dir, '%s.pth' % model_type)

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
                total_loss += loss.item()*k
                total_count += k
                pbar.set_postfix(loss=loss.item(), acc=acc)
        end = time()
        print('训练耗时:%ds, loss=%.3f' % (end - start, total_loss/total_count))

    def eval(self, data_loader, save_best_model=True):
        start = time()
        self.model.eval()

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
                total_loss += loss.item()
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

    def predict(self, image, size, return_prob=True):
        image = square(image, size)
        image = subprocess(image)
        image = torch.tensor(image).unsqueeze(dim=0).to(self.device)
        pred_y = self.model(image)
        pred_y = torch.softmax(pred_y, dim=-1).cpu().numpy()
        pred_indice = np.argmax(pred_y, dim=-1)
        if return_prob:
            return pred_indice, pred_y[pred_indice]
        else:
            return pred_indice

    def model_eval(self):
        self.model.eval()

    def save_model(self, loss, acc):
        if loss < self.min_loss:
            if not os.path.exists(self.save_model_dir):
                os.makedirs(self.save_model_dir)
            self.min_loss = loss
            params = {}
            params['loss'] = loss
            params['acc'] = acc
            params['state_dict'] = self.model.state_dict()
            torch.save(params, self.save_model_path)
            print('min_loss update to %s' % loss)

    def load_model(self, model_path):
        params = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(params['state_dict'])
        self.min_loss = params['loss']
        print('已加载{}路径下的模型，当前效果是loss:{:.3f}, acc:{}'.format(model_path, params['loss'], params['acc']))




if __name__ == '__main__':
    net = PretrainModels('densenet121', 4)
    # print(net.model._modules)
    print(net)
    net.eval()
    torchsummary.summary(net, (3, 256, 256))