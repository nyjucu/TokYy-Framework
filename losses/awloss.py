import torch.nn as nn
import torch.nn.functional as fun
from torch import clamp 

from pytorch_ssim import ssim

def depth_loss( pred, target ):
    return fun.l1_loss( pred, target )

def gradient_loss( pred, target ):
    pred_dy = pred[ :, :, 1:, : ] - pred[ :, :, :-1, : ]
    pred_dx = pred[ :, :, :, 1: ] - pred[ :, :, :, :-1 ]

    target_dy = target[:, :, 1:, :] - target[ :, :, :-1, : ]
    target_dx = target[:, :, :, 1:] - target[ :, :, :, :-1 ]

    loss_dy = fun.l1_loss( pred_dy, target_dy )
    loss_dx = fun.l1_loss( pred_dx, target_dx )

    return loss_dx + loss_dy

def ssim_loss( pred, target ):
    ssim_val = ssim( pred, target )
    ssim_val = clamp( ssim_val, 0, 1 )
    return 1 - ssim_val


class AWLoss( nn.Module ):
    def __init__( self, lambda_depth = 0.1 ):
        super().__init__()

        self.lambda_depth = lambda_depth

    def forward( self, pred, target ):
        l_depth = depth_loss( pred, target )
        l_grad = gradient_loss( pred, target )
        l_ssim = ssim_loss( pred, target )

        return self.lambda_depth * l_depth + l_grad + l_ssim
    