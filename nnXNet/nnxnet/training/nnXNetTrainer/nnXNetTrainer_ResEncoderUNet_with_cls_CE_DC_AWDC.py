import numpy as np
import torch
from nnxnet.training.loss.deep_supervision import DeepSupervisionWrapper
from nnxnet.training.loss.compound_awdice_loss import DC_and_CE_and_AWDC_loss
from nnxnet.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnxnet.training.nnXNetTrainer.nnXNetTrainer_ResEncoderUNet_with_cls import nnXNetTrainer_ResEncoderUNet_with_cls
    
class nnXNetTrainer_ResEncoderUNet_with_cls_CE_DC_AWDC(nnXNetTrainer_ResEncoderUNet_with_cls):

    def _build_loss(self):

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        
        lambda_awdice = 1.0
        lambda_dice = 1.0
        lambda_ce = lambda_dice + lambda_awdice

        seg_ce_class_weights = [1] + self.seg_ce_class_weights
        self.print_to_log_file("seg_ce_class_weights: ", seg_ce_class_weights)
        seg_ce_class_weights_gpu = torch.tensor(seg_ce_class_weights, dtype=torch.float32).to(self.device)

        loss = DC_and_CE_and_AWDC_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'weight': seg_ce_class_weights_gpu},
                                    {'aneurysm_classes':(14, 26), 'aneurysm_weight': 1, 'smooth': 1e-3},
                                    weight_ce=lambda_ce, weight_dice=lambda_dice, weight_awdice=lambda_awdice, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        self.print_to_log_file("lambda_awdice: %s" % str(lambda_awdice))
        self.print_to_log_file("lambda_dice: %s" % str(lambda_dice))
        self.print_to_log_file("lambda_ce: %s" % str(lambda_ce))

        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
    