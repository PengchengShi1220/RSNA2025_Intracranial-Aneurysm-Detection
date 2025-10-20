import os
import pydoc
import csv
import torch
import pydoc
import numpy as np
import warnings
import multiprocessing
from time import time, sleep
from torch import autocast, nn
from torch import distributed as dist
from typing import Union, Tuple, List
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_with_cls import ResEncoderUNet_with_cls
from nnxnet.training.nnXNetTrainer.nnXNetTrainer import nnXNetTrainer
from nnxnet.training.dataloading.nnxnet_dataset import nnXNetDataset
from nnxnet.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnxnet.configuration import ANISO_THRESHOLD, default_num_processes
from nnxnet.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnxnet.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnxnet.inference.predict_from_raw_data_with_cls import nnXNetPredictor
from nnxnet.training.logging.nnxnet_logger import nnXNetLogger
from nnxnet.inference.sliding_window_prediction import compute_gaussian
from nnxnet.paths import nnXNet_preprocessed, nnXNet_results
from nnxnet.training.loss.compound_losses import DC_and_CE_loss
from nnxnet.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnxnet.training.loss.deep_supervision import DeepSupervisionWrapper
from nnxnet.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnxnet.utilities.collate_outputs import collate_outputs
from nnxnet.utilities.file_path_utilities import check_workers_alive_and_busy
from nnxnet.utilities.helpers import empty_cache, dummy_context
from nnxnet.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from sklearn.metrics import accuracy_score, roc_auc_score

class nnXNetTrainer_ResEncoderUNet_with_cls(nnXNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)

        self.cls_task_index = self.configuration_manager.cls_task_index
        self.cls_head_num_classes_list = self.configuration_manager.network_arch_init_kwargs["cls_head_num_classes_list"]
        self.seg_index = self.configuration_manager.seg_index
        self.seg_ce_class_weights = self.configuration_manager.seg_ce_class_weights
        self._init_mapping()

        self.num_cls_task = len(self.cls_task_index)

        pos_weights_list = self.configuration_manager.pos_weights_list

        if pos_weights_list is None:
            self.print_to_log_file("pos_weights_list is None, automatically generating pos_weights based on cls_task_index.")
            self.pos_weights_list = [[1.0] * len(cls_labels) if isinstance(cls_labels[0], list) else [1.0] for cls_labels in self.cls_task_index]
        else:
            self.pos_weights_list = pos_weights_list
        
        self.print_to_log_file(f"Using pos_weights_list: {self.pos_weights_list}")

        self.logger = nnXNetLogger(num_cls_task=self.num_cls_task)
        self.seg_loss_weight = 1.0

        self.save_every = 5
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50

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
            self.cls_loss_list = []
            for i in range(self.num_cls_task):
                pos_weights_tensor = torch.tensor(self.pos_weights_list[i]).to(self.device)
                cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor, reduction='none')
                self.cls_loss_list.append(cls_loss)

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
        network = ResEncoderUNet_with_cls(
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                deep_supervision=enable_deep_supervision,
                **architecture_kwargs
            )
        
        return network
    
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'z': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                } # TODO: need to revisit 15 degree limit
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnxnet for now)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            *rotation_for_DA.values(),
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
    
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
        
        # Ensure target is of long type
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
            output, cls_pred_list = self.network(data)

            target_merged = [self.merge_target_labels(t) for t in target]

            seg_loss = self.loss(output, target_merged)
            total_loss = self.seg_loss_weight * seg_loss

            total_cls_loss = 0

            flat_target = target[0].view(target[0].size(0), -1)

            for t_index, cls_labels in enumerate(self.cls_task_index):
                if not isinstance(cls_labels[0], list):
                    # For a single list of labels, e.g., [[17], [18]]
                    # We create a target tensor of shape [batch_size, 1]
                    cls_target = torch.isin(flat_target, torch.tensor(cls_labels, device=self.device)).any(dim=1, keepdim=True).float()
                else:
                    # For a list of lists (superclasses), e.g., [[17, 18], [19, 20]]
                    # We create a target tensor of shape [batch_size, num_super_classes]
                    cls_target = torch.stack([
                        torch.isin(flat_target, torch.tensor(super_cls_labels, device=self.device)).any(dim=1).float() 
                        for super_cls_labels in cls_labels
                    ], dim=1)
                
                cls_pred_logits = cls_pred_list[t_index]
                
                cls_loss = self.cls_loss_list[t_index](cls_pred_logits, cls_target)
                total_loss += cls_loss.mean()
                total_cls_loss += cls_loss.mean()

            l = total_loss / (self.seg_loss_weight + self.num_cls_task)

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
        return {'loss': l.detach().cpu().numpy(), 'total_cls_loss': total_cls_loss.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        keys = batch['keys']

        validation_dict = {}

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output, cls_pred_list = self.network(data)

            target_merged = [self.merge_target_labels(t) for t in target]
            seg_loss = self.loss(output, target_merged)
            total_loss = self.seg_loss_weight * seg_loss

            flat_target = target[0].view(target[0].size(0), -1)

            total_cls_loss = 0

            for t_index, cls_labels in enumerate(self.cls_task_index):
                if not isinstance(cls_labels[0], list):
                    cls_target = torch.isin(flat_target, torch.tensor(cls_labels, device=self.device)).any(dim=1, keepdim=True).float()
                else:
                    cls_target = torch.stack([
                        torch.isin(flat_target, torch.tensor(super_cls_labels, device=self.device)).any(dim=1).float() 
                        for super_cls_labels in cls_labels
                    ], dim=1)
                
                cls_pred_logits = cls_pred_list[t_index]
                
                cls_loss = self.cls_loss_list[t_index](cls_pred_logits, cls_target)
                total_loss += cls_loss.mean()
                total_cls_loss += cls_loss.mean()
                
                cls_probs = torch.sigmoid(cls_pred_logits)

                validation_dict[f'cls_task_{t_index}_probs'] = cls_probs.detach().cpu().numpy()
                validation_dict[f'cls_task_{t_index}_targets'] = cls_target.detach().cpu().numpy()
                validation_dict[f'cls_task_{t_index}_loss'] = cls_loss.mean().detach().cpu().numpy()

            l = total_loss / (self.seg_loss_weight + self.num_cls_task)

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
        validation_dict['total_cls_loss'] = total_cls_loss.detach().cpu().numpy()
        validation_dict['tp_hard'] = tp_hard
        validation_dict['fp_hard'] = fp_hard
        validation_dict['fn_hard'] = fn_hard

        validation_dict['keys'] = keys

        return validation_dict

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            total_cls_train_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            dist.all_gather_object(total_cls_train_losses_tr, outputs['total_cls_loss'])
            loss_here = np.vstack(losses_tr).mean()
            total_cls_loss_here = np.vstack(total_cls_train_losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            total_cls_loss_here = np.mean(outputs['total_cls_loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

        if 'total_cls_train_losses' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['total_cls_train_losses'] = list()
        self.logger.log('total_cls_train_losses', total_cls_loss_here, self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)
        keys = outputs_collated['keys']

        # Calculate segmentation metrics
        global_dc_per_class = [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        if self.is_ddp:
            world_size = dist.get_world_size()
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()

            total_cls_losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(total_cls_losses_val, outputs_collated['total_cls_loss'])
            total_cls_losses_here = np.vstack(total_cls_losses_val).mean()

            cls_losses_mean = []
            for t_index in range(self.num_cls_task):
                cls_task_losses = [None for _ in range(world_size)]
                dist.all_gather_object(cls_task_losses, [output[f'cls_task_{t_index}_loss'] for output in val_outputs])
                cls_losses_mean.append(np.mean([np.mean(losses) for losses in cls_task_losses]))
        else:
            loss_here = np.mean(outputs_collated['loss'])
            total_cls_losses_here = np.mean(outputs_collated['total_cls_loss'])
            cls_losses_mean = [np.mean([output[f'cls_task_{t_index}_loss'] for output in val_outputs]) 
                            for t_index in range(self.num_cls_task)]
        
        # Prepare CSV data
        csv_data = []
        header = ['Epoch', 'Key', 'Task', 'Loss', 'Classification_Prob', 'Ground_Truth', 'Accuracy', 'AUC', 'Mean_FG_Dice']

        for t_index in range(self.num_cls_task):
            self.logger.log(f'cls_task_{t_index}_loss', cls_losses_mean[t_index], self.current_epoch)
            
            cls_probs_list = [output[f'cls_task_{t_index}_probs'] for output in val_outputs]
            cls_targets_list = [output[f'cls_task_{t_index}_targets'] for output in val_outputs]
            
            if self.is_ddp:
                world_size = dist.get_world_size()
                all_cls_probs = [[] for _ in range(world_size)]
                all_cls_targets = [[] for _ in range(world_size)]
                dist.all_gather_object(all_cls_probs, cls_probs_list)
                dist.all_gather_object(all_cls_targets, cls_targets_list)
                cls_probs = np.concatenate(sum(all_cls_probs, []))
                cls_targets = np.concatenate(sum(all_cls_targets, []))
            else:
                cls_probs = np.concatenate(cls_probs_list)
                cls_targets = np.concatenate(cls_targets_list)

            if cls_targets.ndim == 1:
                cls_targets = cls_targets.reshape(-1, 1)
                cls_probs = cls_probs.reshape(-1, 1)

            auc_list = []
            for i in range(cls_probs.shape[1]):
                if len(np.unique(cls_targets[:, i])) > 1:
                    auc_list.append(roc_auc_score(cls_targets[:, i], cls_probs[:, i]))
                else:
                    auc_list.append(0.5)

            auc = np.mean(auc_list)
            
            cls_preds = (cls_probs > 0.5).astype(int)
            acc = accuracy_score(cls_targets.flatten(), cls_preds.flatten())

            self.logger.log(f'cls_task_{t_index}_acc', acc, self.current_epoch)
            self.logger.log(f'cls_task_{t_index}_auc', auc, self.current_epoch)

            # Add data for CSV, ensuring each key is in a separate row
            for key_idx, key in enumerate(keys):
                for sample_idx in range(len(cls_probs[key_idx])):
                    csv_data.append([
                        self.current_epoch,
                        key,
                        f'task_{t_index}',
                        cls_losses_mean[t_index],
                        cls_probs[key_idx][sample_idx],
                        cls_targets[key_idx][sample_idx],
                        acc,
                        auc,
                        mean_fg_dice
                    ])

        # Log segmentation metrics
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        if 'total_cls_val_losses' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['total_cls_val_losses'] = list()
        self.logger.log('total_cls_val_losses', total_cls_losses_here, self.current_epoch)

        # Write to CSV file (append mode)
        csv_filename = self.output_folder + '/validation_metrics.csv'
        write_header = not os.path.exists(csv_filename)
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(header)
            writer.writerows(csv_data)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('total_cls_train_losses', np.round(self.logger.my_fantastic_logging['total_cls_train_losses'][-1], decimals=4))
        self.print_to_log_file('total_cls_val_losses', np.round(self.logger.my_fantastic_logging['total_cls_val_losses'][-1], decimals=4))

        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        
        for task_i in range(self.num_cls_task):
            self.print_to_log_file(f'cls_task_{task_i}_loss', np.round(self.logger.my_fantastic_logging[f'cls_task_{task_i}_loss'][-1], decimals=4))
            self.print_to_log_file(f'cls_task_{task_i}_acc', np.round(self.logger.my_fantastic_logging[f'cls_task_{task_i}_acc'][-1], decimals=4))
            self.print_to_log_file(f'cls_task_{task_i}_auc', np.round(self.logger.my_fantastic_logging[f'cls_task_{task_i}_auc'][-1], decimals=4))

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
    
    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnX-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = nnXNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnXNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names
            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.keys()):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )

                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnXNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)
                        try:
                            tmp = nnXNetDataset(expected_preprocessed_folder, [k],
                                                 num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                    also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()
    
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
        # self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        # if self.grad_scaler is not None:
        #     if checkpoint['grad_scaler_state'] is not None:
        #         self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
