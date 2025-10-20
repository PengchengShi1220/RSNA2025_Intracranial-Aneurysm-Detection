import torch
import pydoc
import numpy as np
from torch import nn
from typing import Union, Tuple, List
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
from torch import autocast, nn
from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet import ResEncoderUNet

from nnxnet.training.nnXNetTrainer.nnXNetTrainer import nnXNetTrainer
from nnxnet.utilities.get_network_from_plans import get_network_from_plans
from nnxnet.training.loss.compound_losses import DC_and_CE_loss
from nnxnet.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnxnet.training.loss.deep_supervision import DeepSupervisionWrapper
from nnxnet.utilities.helpers import empty_cache, dummy_context
from nnxnet.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels

class nnXNetTrainer_ResEncoderUNet(nnXNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = True

        self.seg_index = self.configuration_manager.seg_index
        self.seg_ce_class_weights = self.configuration_manager.seg_ce_class_weights
        self._init_mapping()

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                len(self.seg_index) + 1,
                self.enable_deep_supervision
            ).to(self.device)

            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
    
    def _build_loss(self):
        seg_ce_class_weights = [1] + self.seg_ce_class_weights
        self.print_to_log_file("seg_ce_class_weights: ", seg_ce_class_weights)
        seg_ce_class_weights_gpu = torch.tensor(seg_ce_class_weights, dtype=torch.float32).to(self.device)
        seg_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'weight': seg_ce_class_weights_gpu}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

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
            seg_loss = DeepSupervisionWrapper(seg_loss, weights)
        return seg_loss
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnX-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        mod.deep_supervision = enabled
        
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:        
        
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
        network = ResEncoderUNet(
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                deep_supervision=enable_deep_supervision,
                **architecture_kwargs
            )
        
        return network
    
    def _init_mapping(self):
        """Precompute label mapping table"""
        if not self.seg_index:
            self.max_label = 0
            self.label_mapping = None
            return
        
        # Find the maximum label value
        self.max_label = max(max(sublist) for sublist in self.seg_index if sublist)
        
        # Create mapping table
        self.label_mapping = torch.zeros(self.max_label + 1, dtype=torch.long).to(self.device)
        
        for new_label, old_labels in enumerate(self.seg_index):
            for old_label in old_labels:
                if old_label <= self.max_label:
                    self.label_mapping[old_label] = new_label + 1

    def merge_target_labels(self, target):
        """
        Merge labels using precomputed mapping table
        
        Args:
            target: Tensor of shape (B, H, W, D) with integer class labels
        
        Returns:
            target_merged: Tensor with merged labels (same shape as target)
        """
        if self.label_mapping is None:
            return target.clone()
        
        target = target.long()
        
        target_merged = torch.zeros_like(target)

        valid_mask = (target <= self.max_label) & (target > 0)
        valid_indices = target[valid_mask].long()
        target_merged[valid_mask] = self.label_mapping[valid_indices]
        
        target_merged[~valid_mask] = target[~valid_mask]

        return target_merged

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)

            target_merged = [self.merge_target_labels(t) for t in target]

            l = self.loss(output, target_merged)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        validation_dict = {}

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)

            target_merged = [self.merge_target_labels(t) for t in target]
            l = self.loss(output, target_merged)

        target = target_merged

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg
        
        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        validation_dict['loss'] = l.detach().cpu().numpy()
        validation_dict['tp_hard'] = tp_hard
        validation_dict['fp_hard'] = fp_hard
        validation_dict['fn_hard'] = fn_hard

        return validation_dict

