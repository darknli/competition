from torchvision import models
from torch import nn
from common.model_zoo.model import Model
from time import time
from common.utils.metric import get_acc
from common.utils.loss import cross_entroy, smoth_l1
from common.utils.subprocess import *
from tqdm import tqdm
import os
from numpy import inf
import torchsummary
import torch

class PretrainBoxModels(nn.Module):
    def __init__(self, model_type, num_nc, num_classes):
        super(PretrainBoxModels, self).__init__()
        if model_type == 'inception_v3_google':
            self.model = models.inception_v3(pretrained=True, aux_logits=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_nc)
        elif model_type == 'resnet50':
            self.model = models.resnet50(pretrained=True,)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_nc)
        elif model_type == 'resnet152':
            self.model = models.resnet152(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_nc)
        elif model_type == 'resnext101_32x8d':
            self.model = models.resnext101_32x8d(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_nc)
        elif model_type == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_nc)
        elif model_type == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_nc)
        elif model_type == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=True)
            self.model.classifier._modules['1'] = nn.Linear(self.model.classifier._modules['1'].in_features, num_nc)
        else:
            raise ValueError('PretrainModels沒有這個模型！ ')
        self.model_type = model_type
        self.cls = nn.Linear(num_nc, num_classes)
        self.reg = nn.Linear(num_nc, 4)

    def forward(self, x):
        y = self.model(x)
        # pred_cls = self.cls(y)
        pred_reg = self.reg(y)
        return y, pred_reg


class BoxModel:
    def __init__(self, model_type, num_classes, num_nc, device, save_model_dir='model', opt_mode='adam', learning_rate=10e-2):
        super().__init__(model_type, num_classes, device, save_model_dir, opt_mode, learning_rate)
        self.model = PretrainBoxModels(model_type, num_nc, num_classes)
        self.model = self.model.to(device)
        # torchsummary.summary(self.model, input_size=(3, 256, 256))
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

        cls_factor = 1
        box_factor = 1
        total_loss = 0
        total_count = 0
        with tqdm(data_loader) as pbar:
            for image, label, bbox in pbar:
                self.optimizer.zero_grad()
                image = image.to(self.device)
                label = label.to(self.device)
                bbox = bbox.to(self.device)
                pred_y, pred_box = self.model(image)
                acc = get_acc(pred_y, label).item()
                cls_loss, _ = cross_entroy(pred_y, label, 0.8)
                # box_loss, _ = smoth_l1(pred_box, bbox, 1)
                # loss = cls_loss * cls_factor + box_loss * box_factor
                loss = cls_loss * cls_factor
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_count += 1
                pbar.set_postfix(loss=loss.item(), acc=acc)
        end = time()
        print('训练耗时:%ds, loss=%.3f' % (end - start, total_loss/total_count))

    def set_num_fintune_layers(self, num_layers):
        layers = [param for param in self.model.parameters()]
        if 0 < num_layers < 1:
            num_layers = int(len(layers) * 0.5)
        for param in layers[:-num_layers]:
            param.requires_grad = False

    def eval(self, data_loader, save_best_model=True):
        start = time()
        self.model.eval()
        total_loss = 0
        total_count = 0
        total_pred_result = []
        total_true_result = []
        with torch.no_grad():
            for image, label, box in data_loader:
                self.optimizer.zero_grad()
                image = image.to(self.device)
                label = label.to(self.device)
                pred_y, pred_box = self.model(image)
                cls_loss, _ = cross_entroy(pred_y, label, 1)
                total_loss += cls_loss.item()
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

    def predict(self, image, size, return_prob=False):
        image = square(image, size)
        image = subprocess(image)
        image = torch.tensor(image).unsqueeze(dim=0).to(self.device)
        pred_y = self.model(image)
        pred_y = torch.softmax(pred_y, dim=-1).cpu().numpy()
        pred_indice = np.argmax(pred_y, -1)
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
    import torch
    model = PretrainBoxModels('inception_v3_google', 128, 5).cuda()
    x = torch.rand((10, 3, 256, 256)).cuda()
    y = model(x)
    print(y.shape)
