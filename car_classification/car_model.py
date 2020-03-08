from efficientnet_pytorch import EfficientNet
from torchvision import models
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import os
from car_classification.utils import *
import numpy as np
from tqdm import tqdm
import torchsummary


class pretrainNet(nn.Module):
    def __init__(self, model_type, num_classes, pretrained=True):
        super(pretrainNet, self).__init__()
        if model_type == "wide_resnet50_2":
            model = models.wide_resnet50_2(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        if model_type == "wide_resnet101_2":
            model = models.wide_resnet101_2(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_type == 'EfficientNet':
            model = EfficientNet.from_pretrained('efficientnet-b0')
            model._fc = nn.Linear(1280, num_classes)
        elif model_type == 'inception_v3_google':
            model = models.inception_v3(pretrained=True, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == 'resnet50':
            model = models.resnet50(pretrained=True, )
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == 'resnet152':
            model = models.resnet152(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == 'resnext101_32x8d':
            model = models.resnext101_32x8d(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == 'densenet121':
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features,
                                              num_classes)
        elif model_type == 'densenet201':
            model = models.densenet201(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features,
                                              num_classes)
        elif model_type == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier._modules['1'] = nn.Linear(model.classifier._modules['1'].in_features,
                                                            num_classes)
        elif model_type == 'mnasnet0_5':
            model = models.mnasnet0_5(pretrained=True)
            model.classifier._modules['1'] = nn.Linear(model.classifier._modules['1'].in_features,
                                                            num_classes)
        elif model_type == 'mnasnet0_75':
            model = models.mnasnet0_5(pretrained=True)
            model.classifier._modules['1'] = nn.Linear(model.classifier._modules['1'].in_features,
                                                            num_classes)
        elif model_type == 'squeezenet1_0':
            model = models.squeezenet1_0(pretrained=True)
            model.classifier.final_conv = nn.Conv2d(1000, num_classes, kernel_size=1)
        elif model_type == 'vgg16':
            model = models.vgg16(pretrained=True)
            model.classifier._modules['6'] = nn.Linear(model.classifier._modules['6'].in_features,
                                                            num_classes)
        else:
            raise ValueError("不支持这个模型")
        self.model = model

    def forward(self, img):
        out = self.model(img)
        return out


class Model:
    def __init__(self, model_type, num_classes, criterion, pretrained=True, device='cpu', dir='models', prefix="", suffix=""):
        self.model_type = model_type
        self.model = pretrainNet(model_type, num_classes, pretrained).to(device)
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.001)
        self.criterion = criterion
        self.device = device
        self.min_loss = np.inf
        self.set_save_config(dir, prefix, suffix)
        torchsummary.summary(self.model, (3, 256, 256))


    def fit(self, data):
        total_loss = AverageMeter('train_loss')
        total_acc = AverageMeter('acc')
        self.model.train()

        with tqdm(data) as pbar:
            for input, target in pbar:
                input = input.to(self.device)
                target = target.to(self.device)

                output = self.model(input)
                loss = self.criterion(output, target)
                acc = accuracy(output, target, topk=(1,))[0]
                total_loss.update(loss.item(), len(input))
                total_acc.update(acc, len(input))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=str(total_loss), top1=str(total_acc))

        print('训练集 loss={}, acc={}'.format(total_loss, total_acc))

    def predict(self, data, tta=5):
        self.model.eval()

        test_pred_tta = None
        for _ in range(tta):
            test_pred = []
            with torch.no_grad():
                for input in tqdm(data):
                    input = input.to(self.device)
                    output = self.model(input)
                    output = output.data.cpu().numpy()
                    test_pred.append(output)
            test_pred = np.vstack(test_pred)

            if test_pred_tta is None:
                test_pred_tta = test_pred
            else:
                test_pred_tta += test_pred
        return test_pred_tta

    def validate(self, val_loader, save_best_model=True):
        total_loss = AverageMeter('val_loss')
        total_acc = AverageMeter('acc')

        self.model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input = input.cuda()
                target = target.cuda()

                output = self.model(input)
                loss = self.criterion(output, target)

                acc = accuracy(output, target, topk=(1,))[0]
                total_loss.update(loss.item(), len(input))
                total_acc.update(acc, len(input))
        print('验证集 loss={}, acc={}'.format(total_loss, total_acc))
        if save_best_model and total_loss.avg < self.min_loss:
            self.min_loss = total_loss.avg
            print('min_loss update to %f' % self.min_loss)
            self.save_model(self.dir, self.prefix, self.suffix)

    def save_model(self, dir='models', prefix="", suffix="", loss=None):
        if not os.path.exists(dir):
            os.makedirs(dir)

        if prefix != "":
            prefix += "_"
        if suffix != "":
            suffix = "_"+suffix

        if loss is None:
            loss = self.min_loss
        param = {
            "weights": self.model.state_dict(),
            "loss": loss
        }
        torch.save(param, '{}/{}{}{}.pth'.format(dir, prefix,  self.model_type, suffix))

    def load_model(self, path):
        params = torch.load(path, map_location=self.device)
        self.model.load_state_dict(params['weights'])
        print('加载模型成功，loss ', params["loss"])

    def set_save_config(self, dir='models', prefix="", suffix=""):
        self.dir = dir
        self.prefix = prefix
        self.suffix = suffix


