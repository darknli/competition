import torch
from torch.nn import functional

def cross_entroy(pred_y, true_y, ohem_rate):
    k = int(pred_y.shape[0] * ohem_rate)
    if k == 0:
        return torch.tensor(0.0).to(pred_y.device), 0
    loss = functional.cross_entropy(pred_y, true_y, reduction='none')
    loss = torch.topk(loss, k, largest=True, sorted=True)[0]
    loss = torch.mean(loss)
    return loss, k

def smoth_l1(pred_box, target_box, ohem_rate):
    k = int(pred_box.shape[0] * ohem_rate)
    if k == 0:
        return torch.tensor(0.0).to(pred_box.device), 0
    loss = functional.smooth_l1_loss(pred_box, target_box, reduction='none')
    loss = torch.sum(loss, dim=1)
    loss = torch.topk(loss, k, largest=True, sorted=True)[0]
    loss = torch.mean(loss)
    return loss, k