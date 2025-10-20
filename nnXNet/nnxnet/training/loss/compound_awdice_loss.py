import torch
from nnxnet.training.loss.dice import SoftDiceLoss
from nnxnet.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnxnet.training.loss.awdice_loss import AneurysmWeightedDiceLoss
from nnxnet.utilities.helpers import softmax_helper_dim1
from torch import nn

class DC_and_CE_and_AWDC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, awdc_kwargs, weight_ce=1, weight_dice=1, weight_awdice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE, Dice, and Aneurysm-Weighted Dice do not need to sum to one. You can set whatever you want.
        
        Args:
            soft_dice_kwargs (dict): Keyword arguments for SoftDiceLoss.
            ce_kwargs (dict): Keyword arguments for RobustCrossEntropyLoss.
            awdc_kwargs (dict): Keyword arguments for AneurysmWeightedDiceLoss, including aneurysm_weight.
            weight_ce (float): Weight for cross-entropy loss.
            weight_dice (float): Weight for standard Dice loss.
            weight_awdice (float): Weight for aneurysm-weighted Dice loss.
            ignore_label (int, optional): Label to ignore in loss computation.
            dice_class: Class for Dice loss (default: SoftDiceLoss).
        """
        super(DC_and_CE_and_AWDC_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_awdice = weight_awdice
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.awdice = AneurysmWeightedDiceLoss(**awdc_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        Forward pass for the combined loss.
        
        Args:
            net_output (torch.Tensor): Network output with shape (b, c, x, y[, z]).
            target (torch.Tensor): Ground truth labels with shape (b, 1, x, y[, z]).
        
        Returns:
            torch.Tensor: Combined loss (weighted sum of CE, Dice, and Aneurysm-Weighted Dice losses).
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        awdice_loss = self.awdice(net_output, target) if self.weight_awdice != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_awdice * awdice_loss
        return result