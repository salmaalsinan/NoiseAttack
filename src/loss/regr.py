import torch
from math import exp
import torch.nn.functional as F


def l1(pred_y, y):
    l1 = torch.mean(torch.abs(pred_y-y))
    return l1

def scale_invariant_loss(pred_y, y, stability=1e-4):
    _, C, H, W = pred_y.shape
    pred_y = torch.clamp(pred_y, min = stability)
    y = torch.clamp(y, min = stability)
    d = torch.log(pred_y.view(-1,C,H*W)) - torch.log(y.view(-1,C,H*W))
    term_1 = torch.pow(d,2).mean(dim=-1)
    term_2 = (torch.pow(d.sum(dim=-1),2)/(2*((H*W)**2)))
    return (term_1 - term_2).mean()

def huber_loss(pred_y, y, reduce='mean'):
    return F.smooth_l1_loss(pred_y, y, reduce=reduce)

