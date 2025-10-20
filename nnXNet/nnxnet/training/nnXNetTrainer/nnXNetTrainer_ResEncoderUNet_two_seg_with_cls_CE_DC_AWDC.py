import torch
import numpy as np
from nnxnet.training.loss.deep_supervision import DeepSupervisionWrapper
from nnxnet.training.loss.compound_awdice_loss import DC_and_CE_and_AWDC_loss
from nnxnet.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnxnet.training.nnXNetTrainer.nnXNetTrainer_ResEncoderUNet_two_seg_with_cls import nnXNetTrainer_ResEncoderUNet_two_seg_with_cls
    
class nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_CE_DC_AWDC(nnXNetTrainer_ResEncoderUNet_two_seg_with_cls):

    def _build_loss(self):
        seg_ce_class_weights_1 = [1] + self.seg_ce_class_weights_1
        seg_ce_class_weights_2 = [1] + self.seg_ce_class_weights_2
        self.print_to_log_file("seg_ce_class_weights_1: ", seg_ce_class_weights_1)
        self.print_to_log_file("seg_ce_class_weights_2: ", seg_ce_class_weights_2)
        seg_ce_class_weights_1_gpu = torch.tensor(seg_ce_class_weights_1, dtype=torch.float32).to(self.device)
        seg_ce_class_weights_2_gpu = torch.tensor(seg_ce_class_weights_2, dtype=torch.float32).to(self.device)

        lambda_awdice = 1.0
        lambda_dice = 1.0
        lambda_ce = lambda_dice + lambda_awdice

        seg_loss_1 = DC_and_CE_and_AWDC_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'weight': seg_ce_class_weights_1_gpu},
                                    {'aneurysm_classes':(14, 26), 'aneurysm_weight': 1, 'smooth': 1e-3},
                                    weight_ce=lambda_ce, weight_dice=lambda_dice, weight_awdice=lambda_awdice, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        
        seg_loss_2 = DC_and_CE_and_AWDC_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'weight': seg_ce_class_weights_2_gpu},
                                    {'aneurysm_classes':(14, 26), 'aneurysm_weight': 1, 'smooth': 1e-3},
                                    weight_ce=lambda_ce, weight_dice=lambda_dice, weight_awdice=lambda_awdice, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        
        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            seg_loss_1 = DeepSupervisionWrapper(seg_loss_1, weights)
            seg_loss_2 = DeepSupervisionWrapper(seg_loss_2, weights)
        return seg_loss_1, seg_loss_2
    