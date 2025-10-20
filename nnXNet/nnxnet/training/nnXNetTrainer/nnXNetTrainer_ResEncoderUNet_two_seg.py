import torch
import pydoc
import numpy as np
from time import time, sleep
from torch import nn
from typing import Union, Tuple, List
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
from torch import autocast, nn
from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg import ResEncoderUNet_two_seg

from nnxnet.training.nnXNetTrainer.nnXNetTrainer import nnXNetTrainer
from nnxnet.utilities.get_network_from_plans import get_network_from_plans
from nnxnet.training.logging.nnxnet_logger import nnXNetLogger
from nnxnet.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnxnet.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnxnet.training.loss.deep_supervision import DeepSupervisionWrapper
from nnxnet.training.loss.dice import get_tp_fp_fn_tn
from nnxnet.utilities.collate_outputs import collate_outputs
from nnxnet.utilities.helpers import empty_cache, dummy_context
from nnxnet.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from batchgenerators.utilities.file_and_folder_operations import join

class nnXNetTrainer_ResEncoderUNet_two_seg(nnXNetTrainer):
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

        self.seg_index_1 = self.configuration_manager.seg_index_1
        self.seg_index_2 = self.configuration_manager.seg_index_2
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
                len(self.seg_index_1) + 1,
                len(self.seg_index_2) + 1,
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

            self.seg_1_loss, self.seg_2_loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
    
    def _build_loss(self):
        seg_1_ce_class_weights = [1, 1, 3, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 10, 5]
        seg_1_ce_class_weights = torch.tensor(seg_1_ce_class_weights, dtype=torch.float32).to(self.device)
        seg_1_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'weight': seg_1_ce_class_weights}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        seg_2_ce_class_weights = [1, 1, 3, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 10]
        seg_2_ce_class_weights = torch.tensor(seg_2_ce_class_weights, dtype=torch.float32).to(self.device)
        seg_2_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'weight': seg_2_ce_class_weights}, weight_ce=1, weight_dice=1,
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
            seg_1_loss = DeepSupervisionWrapper(seg_1_loss, weights)
            seg_2_loss = DeepSupervisionWrapper(seg_2_loss, weights)
        return seg_1_loss, seg_2_loss
    
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
                                   num_output_channels_1: int,
                                   num_output_channels_2: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:        
        
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
        network = ResEncoderUNet_two_seg(
                in_channels=num_input_channels,
                out_channels_1=num_output_channels_1,
                out_channels_2=num_output_channels_2,
                deep_supervision=enable_deep_supervision,
                **architecture_kwargs
            )
        
        return network
    
    def _init_mapping(self):
        """Precompute label mapping tables for multiple segmentation heads"""
        # Get all seg_index attributes
        seg_index_attrs = [attr for attr in dir(self) if attr.startswith('seg_index_')]
        
        if not seg_index_attrs:
            self.max_label = 0
            self.label_mappings = None
            return
        
        # Store all mapping tables
        self.label_mappings = {}
        
        # Create mapping for each segmentation head
        for seg_attr in seg_index_attrs:
            seg_index = getattr(self, seg_attr)
            if not seg_index:
                continue
            
            # Find maximum label value
            max_label = max(max(sublist) for sublist in seg_index if sublist)
            
            # Create mapping table
            label_mapping = torch.zeros(max_label + 1, dtype=torch.long).to(self.device)
            
            for new_label, old_labels in enumerate(seg_index):
                for old_label in old_labels:
                    if old_label <= max_label:
                        label_mapping[old_label] = new_label + 1
            
            # Store mapping table using segmentation head name as key
            self.label_mappings[seg_attr] = {
                'mapping': label_mapping,
                'max_label': max_label
            }

    def merge_target_labels(self, target, head_name=None):
        """
        Merge labels using precomputed mapping table
        
        Args:
            target: Tensor of shape (B, H, W, D) with integer class labels
            head_name: Specify which segmentation head's mapping table to use, 
                    if None use the first one
        
        Returns:
            target_merged: Tensor with merged labels (same shape as target)
        """
        if self.label_mappings is None or not self.label_mappings:
            return target.clone()
        
        # If no segmentation head specified, use the first one
        if head_name is None:
            head_name = list(self.label_mappings.keys())[0]
        
        if head_name not in self.label_mappings:
            raise ValueError(f"Segmentation head '{head_name}' does not exist")
        
        mapping_info = self.label_mappings[head_name]
        label_mapping = mapping_info['mapping']
        max_label = mapping_info['max_label']
        
        target = target.long()
        
        target_merged = torch.zeros_like(target)
        valid_mask = (target <= max_label) & (target > 0)
        valid_indices = target[valid_mask].long()
        target_merged[valid_mask] = label_mapping[valid_indices]
        
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
            output_1, output_2 = self.network(data)

            target_merged_1 = [self.merge_target_labels(t, head_name='seg_index_1') for t in target]
            target_merged_2 = [self.merge_target_labels(t, head_name='seg_index_2') for t in target]
            
            l_1 = self.seg_1_loss(output_1, target_merged_1)
            l_2 = self.seg_2_loss(output_2, target_merged_2)
            l = l_1 + l_2

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
            output_1, output_2 = self.network(data)

            target_merged_1 = [self.merge_target_labels(t, head_name='seg_index_1') for t in target]
            target_merged_2 = [self.merge_target_labels(t, head_name='seg_index_2') for t in target]

            l_1 = self.seg_1_loss(output_1, target_merged_1)
            l_2 = self.seg_2_loss(output_2, target_merged_2)
            l = l_1 + l_2

        target_1 = target_merged_1
        target_2 = target_merged_2

        if self.enable_deep_supervision:
            output_1 = output_1[0]
            output_2 = output_2[0]
            target_1 = target_1[0]
            target_2 = target_2[0]

        axes = [0] + list(range(2, output_1.ndim))
        mask = None

        targets = [target_1, target_2]
        outputs = [output_1, output_2]
        target_names = ['1', '2']

        for i, (output, target, name_suffix) in enumerate(zip(outputs, targets, target_names)):
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg
            
            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
            
            validation_dict[f'tp_hard_{name_suffix}'] = tp_hard
            validation_dict[f'fp_hard_{name_suffix}'] = fp_hard
            validation_dict[f'fn_hard_{name_suffix}'] = fn_hard

        validation_dict['loss'] = l.detach().cpu().numpy()

        return validation_dict
    
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        
        tp_1 = np.sum(outputs_collated['tp_hard_1'], 0)
        fp_1 = np.sum(outputs_collated['fp_hard_1'], 0)
        fn_1 = np.sum(outputs_collated['fn_hard_1'], 0)
        
        tp_2 = np.sum(outputs_collated['tp_hard_2'], 0)
        fp_2 = np.sum(outputs_collated['fp_hard_2'], 0)
        fn_2 = np.sum(outputs_collated['fn_hard_2'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])
 
        # Calculate and record segmentation metrics for the first target
        global_dc_per_class_1 = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp_1, fp_1, fn_1)]]
        mean_fg_dice_1 = np.nanmean(global_dc_per_class_1)
        self.logger.log('mean_fg_dice_1', mean_fg_dice_1, self.current_epoch)
        self.logger.log('dice_per_class_or_region_1', global_dc_per_class_1, self.current_epoch)

        # Calculate and record segmentation metrics for the second target
        global_dc_per_class_2 = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp_2, fp_2, fn_2)]]
        mean_fg_dice_2 = np.nanmean(global_dc_per_class_2)
        self.logger.log('mean_fg_dice_2', mean_fg_dice_2, self.current_epoch)
        self.logger.log('dice_per_class_or_region_2', global_dc_per_class_2, self.current_epoch)

        self.logger.log('mean_fg_dice', mean_fg_dice_2, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class_2, self.current_epoch)

        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        
        self.print_to_log_file('Pseudo dice 1', [np.round(i, decimals=4) for i in
                                            self.logger.my_fantastic_logging['dice_per_class_or_region_1'][-1]])
        self.print_to_log_file('Pseudo dice 2', [np.round(i, decimals=4) for i in
                                            self.logger.my_fantastic_logging['dice_per_class_or_region_2'][-1]])
        
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            model_dict = self.network.state_dict()
        
            matched_state_dict = {k: v for k, v in new_state_dict.items() 
                                if k in model_dict and model_dict[k].shape == v.shape}
            
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(matched_state_dict, strict=False)
            else:
                self.network.load_state_dict(matched_state_dict, strict=False)