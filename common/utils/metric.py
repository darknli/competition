import torch

def get_acc(pred_y, true_y):
    pred_y = torch.softmax(pred_y, dim=-1)
    pred_y = torch.argmax(pred_y, dim=-1)
    return torch.sum(pred_y == true_y).float()/true_y.shape[0]

# def get_precision(pred_y, true_y):
#     pred_y = torch.softmax(pred_y, dim=-1)
