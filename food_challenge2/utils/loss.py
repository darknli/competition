import torch
from torch.nn import functional

def cross_entroy(pred_y, true_y, ohem_ration):
    k = int(pred_y.shape[0] * ohem_ration)
    if k == 0:
        return torch.tensor(0.0).to(pred_y.device), 0
    loss = functional.cross_entropy(pred_y, true_y, reduction='none')
    loss = torch.topk(loss, k, largest=True, sorted=True)[0]
    loss = torch.mean(loss)
    return loss, k