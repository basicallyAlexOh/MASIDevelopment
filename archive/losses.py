import numpy as np
from torch.nn import CrossEntropyLoss

# class DiceLoss(nn.Module):

def dice_coeff(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """
    seg = seg.flatten()
    gt = gt.flatten()
    seg[seg > ratio] = np.float32(1)
    seg[seg < ratio] = np.float32(0)
    dice = float(2 * (gt * seg).sum())/float(gt.sum() + seg.sum())
    return dice

def cross_entropy_loss(pred, label):
    loss = CrossEntropyLoss()
    loss(pred, label)
    return loss
