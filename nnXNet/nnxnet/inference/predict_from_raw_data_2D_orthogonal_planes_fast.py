import os
from typing import Tuple, Union, List, Optional
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, save_json
import nnxnet
from torch._dynamo import OptimizedModule
from nnxnet.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnxnet.utilities.find_class_by_name import recursive_find_python_class
from nnxnet.utilities.helpers import empty_cache, dummy_context
from nnxnet.utilities.json_export import recursive_fix_for_json_export
from nnxnet.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnxnet.utilities.label_handling.label_handling import determine_num_input_channels
import SimpleITK as sitk
from nnxnet.imageio.simpleitk_reader_writer import SimpleITKIO
from nnxnet.preprocessing.resampling.resample_torch import resample_torch_simple
from multiprocessing import Process, Queue

class nnXNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 allow_tqdm: bool = True):
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnXNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnxnet.__path__[0], "training", "nnXNetTrainer"),
                                                    trainer_name, 'nnxnet.training.nnXNetTrainer')
        
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnxnet.training.nnXNetTrainer. '
                               f'Please place it there (in any .py file)!')

        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnXNet_compile' in os.environ.keys()) and (os.environ['nnXNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...], patch_size, tile_step_size):
        slicers = []
        is_2d_plus_channels = len(patch_size) < len(image_size)

        if is_2d_plus_channels:
            assert len(patch_size) == len(image_size) - 1, "Patch size must be one dimension shorter than image size for 2D+C."
            steps = compute_steps_for_sliding_window(image_size[1:], patch_size, tile_step_size)
            slicers = [tuple([slice(None), d, *[slice(si, si + ti) for si, ti in zip((sx, sy), patch_size)]])
                       for d in range(image_size[0]) for sx in steps[0] for sy in steps[1]]
        else:
            steps = compute_steps_for_sliding_window(image_size, patch_size, tile_step_size)
            slicers = [tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), patch_size)]])
                       for sx in steps[0] for sy in steps[1] for sz in steps[2]]

        return slicers
    
    def initialize_network_and_gaussian(self):
        """
        Initializes the network by loading parameters, setting it to evaluation mode,
        moving it to the specified device, and precomputing the Gaussian weights if needed.

        Args:
            self: The instance containing network, list_of_parameters, device,
                configuration_manager, and use_gaussian attributes.
        """
        # Load network parameters and set to evaluation mode
        if isinstance(self.network, OptimizedModule):
            self.network._orig_mod.load_state_dict(self.list_of_parameters[0])
        else:
            self.network.load_state_dict(self.list_of_parameters[0])

        self.network.to(self.device).eval()
        empty_cache(self.device)

        # Precompute Gaussian once
        self.gaussian = compute_gaussian(
            tuple(self.configuration_manager.patch_size),
            sigma_scale=1. / 8,
            value_scaling_factor=10,
            device=self.device
        ) if self.use_gaussian else None

        # return self.network, self.gaussian

    def predict_from_multi_axial_slices(self, input_image_np: np.ndarray, original_spacing: np.ndarray, target_spacing: np.ndarray, max_batch_size=16, mask_return=False):
        """
        Performs inference on a 3D image by taking 2D slices from Z, Y, and X axes,
        processing them, and fusing the results to get a single 3D bounding box prediction.

        Args:
            input_image_np (np.ndarray): The input 3D image as a NumPy array (C, Z, Y, X).
            original_spacing (np.ndarray): The original spacing.
            target_spacing (np.ndarray): The target spacing for resampling (e.g., [1, 0.6, 0.4492]).

        Returns:
            np.ndarray: A 3D NumPy array of shape (Z, Y, X) representing the predicted bounding box mask.
        """
        c, z_dim, y_dim, x_dim = input_image_np.shape

        all_patches = []
        all_slicers_info = []

        # --- 收集并预处理所有轴向的patches ---
        for axis_name in ['Z', 'Y', 'X']:
            slices = input_image_np[0, :, :, :]

            if axis_name == 'Z':
                spacings = np.array([original_spacing[1], original_spacing[0]])

            elif axis_name == 'Y':
                slices = np.transpose(slices, (1, 0, 2))
                spacings = np.array([original_spacing[2], original_spacing[0]])
            else:  # axis_name == 'X'
                slices = np.transpose(slices, (2, 0, 1))
                spacings = np.array([original_spacing[2], original_spacing[1]])

            # Optimized indices selection: vectorized non-empty check
            candidates = [slices.shape[0] // 2, slices.shape[0] // 4, slices.shape[0] * 3 // 4]
            non_empty_indices = [idx for idx in candidates if np.any(slices[idx])]

            for idx in non_empty_indices:
                image_slice_np = slices[idx, :, :][None][None]

                # 1. 归一化
                image_slice_np = (image_slice_np - image_slice_np.mean()) / (image_slice_np.std().clip(1e-8))

                # 2. 重采样
                current_spacing = np.concatenate([np.array([1]), spacings])
                dst_shape = [1, int(round(image_slice_np.shape[2] * current_spacing[1] / target_spacing[1])),
                            int(round(image_slice_np.shape[3] * current_spacing[2] / target_spacing[2]))]

                image_resized = resample_torch_simple(
                    torch.from_numpy(image_slice_np).float().to(self.device),
                    dst_shape,
                    is_seg=False,
                    device=self.device,
                    mode='trilinear'
                )

                # 3. 填充和切片
                padded_image, slicer_revert_padding = pad_nd_image(image_resized, self.configuration_manager.patch_size, 'constant', {'value': 0}, True, None)
                slicers = self._internal_get_sliding_window_slicers(padded_image.shape[1:], self.configuration_manager.patch_size, self.tile_step_size)

                for i, sl in enumerate(slicers):
                    patch = padded_image[sl]  # Keep on GPU, no .cpu()
                    all_patches.append(patch)
                    all_slicers_info.append({
                        'axis': axis_name,
                        'idx': idx,
                        'slicer': sl,
                        'padded_shape': padded_image.shape,
                        'original_shape': image_slice_np.shape,
                        'revert_padding': slicer_revert_padding
                    })

        # --- 批量推理 ---
        if not all_patches:
            if mask_return:
                final_bbox_mask = np.zeros(input_image_np.shape[1:], dtype=np.uint8)
                return final_bbox_mask
            else:
                return 0, z_dim, 0, y_dim, 0, x_dim

        all_predictions = []
        for i in range(0, len(all_patches), max_batch_size):
            batch_patches = all_patches[i:i + max_batch_size]
            batch_tensor = torch.stack(batch_patches, dim=0).to(self.device)
            
            with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
                with torch.no_grad():
                    batch_pred = self.network(batch_tensor)
                    all_predictions.append(batch_pred)
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        full_batch_predictions = torch.cat(all_predictions, dim=0) if all_predictions else torch.tensor([])

        # --- 分配结果并后处理 ---
        aggregated_results = {}
        for info in all_slicers_info:
            key = (info['axis'], info['idx'])
            if key not in aggregated_results:
                num_classes = 2  # Assuming binary segmentation (vessel vs background)
                shape = info['padded_shape']
                aggregated_results[key] = {
                    'logits': torch.zeros((num_classes, *shape[1:]), dtype=torch.half, device=self.device),
                    'n_predictions': torch.zeros(shape[1:], dtype=torch.half, device=self.device),
                    'info': info
                }

        for i, prediction in enumerate(full_batch_predictions):
            info = all_slicers_info[i]
            key = (info['axis'], info['idx'])
            if self.use_gaussian:
                prediction = prediction * self.gaussian
                weight = self.gaussian
            else:
                weight = 1.0

            sl = info['slicer']
            aggregated_results[key]['logits'][sl] += prediction
            aggregated_results[key]['n_predictions'][sl[1:]] += weight

        predictions = []
        for key, result in aggregated_results.items():
            info = result['info']
            logits = result['logits']
            n_predictions = result['n_predictions']

            # Inplace division with unsqueeze for broadcasting
            logits.div_(n_predictions.unsqueeze(0))
            logits = logits[(slice(None), *info['revert_padding'][1:])]

            logits_resampled = resample_torch_simple(
                logits,
                info['original_shape'][1:],
                is_seg=False,
                device=self.device,
                mode='trilinear'
            )

            pred_array = logits_resampled.cpu().argmax(0).numpy()
            if np.any(pred_array > 0):
                predictions.append((info['axis'], 'center' if info['idx'] == input_image_np.shape[1] // 2 else 'sub', pred_array, info['idx']))

        # --- 融合所有预测结果并生成3D Bounding Box ---
        if not predictions:
            if mask_return:
                final_bbox_mask = np.zeros(input_image_np.shape[1:], dtype=np.uint8)
                return final_bbox_mask
            else:
                return 0, z_dim, 0, y_dim, 0, x_dim

        # Vectorized bbox collection (original loop is fine, but for clarity keep similar; no major change needed as loop is short)
        final_z_min, final_z_max = [], []
        final_y_min, final_y_max = [], []
        final_x_min, final_x_max = [], []

        for axis, _, pred_slice, idx in predictions:
            if axis == 'Z':
                z_min, z_max = idx, idx + 1
                y_coords, x_coords = np.where(pred_slice.squeeze() > 0)
                if len(y_coords) > 0:
                    y_min, y_max = y_coords.min(), y_coords.max() + 1
                    x_min, x_max = x_coords.min(), x_coords.max() + 1
                    final_y_min.append(y_min); final_y_max.append(y_max)
                    final_x_min.append(x_min); final_x_max.append(x_max)
            elif axis == 'Y':
                y_min, y_max = idx, idx + 1
                z_coords, x_coords = np.where(pred_slice.squeeze() > 0)
                if len(z_coords) > 0:
                    z_min, z_max = z_coords.min(), z_coords.max() + 1
                    x_min, x_max = x_coords.min(), x_coords.max() + 1
                    final_z_min.append(z_min); final_z_max.append(z_max)
                    final_x_min.append(x_min); final_x_max.append(x_max)
            elif axis == 'X':
                x_min, x_max = idx, idx + 1
                z_coords, y_coords = np.where(pred_slice.squeeze() > 0)
                if len(z_coords) > 0:
                    z_min, z_max = z_coords.min(), z_coords.max() + 1
                    y_min, y_max = y_coords.min(), y_coords.max() + 1
                    final_z_min.append(z_min); final_z_max.append(z_max)
                    final_y_min.append(y_min); final_y_max.append(y_max)

        z_min_final = int(np.mean(final_z_min)) if final_z_min else 0
        z_max_final = int(np.mean(final_z_max)) if final_z_max else z_dim
        y_min_final = int(np.mean(final_y_min)) if final_y_min else 0
        y_max_final = int(np.mean(final_y_max)) if final_y_max else y_dim
        x_min_final = int(np.mean(final_x_min)) if final_x_min else 0
        x_max_final = int(np.mean(final_x_max)) if final_x_max else x_dim

        if mask_return:
            final_bbox_mask = np.zeros(input_image_np.shape[1:], dtype=np.uint8)
            final_bbox_mask[z_min_final:z_max_final, y_min_final:y_max_final, x_min_final:x_max_final] = 1
            return final_bbox_mask
        else:
            return z_min_final, z_max_final, y_min_final, y_max_final, x_min_final, x_max_final

    # def predict_from_multi_axial_slices(self, input_image_np: np.ndarray, original_spacing: np.ndarray, target_spacing: np.ndarray):
    #     """
    #     Performs inference on a 3D image by taking 2D slices from Z, Y, and X axes,
    #     processing them, and fusing the results to get a single 3D bounding box prediction.

    #     Args:
    #         input_image_np (np.ndarray): The input 3D image as a NumPy array (C, Z, Y, X).
    #         original_spacing (np.ndarray): The original spacing.
    #         target_spacing (np.ndarray): The target spacing for resampling (e.g., [1, 0.6, 0.4492]).

    #     Returns:
    #         np.ndarray: A 3D NumPy array of shape (Z, Y, X) representing the predicted bounding box mask.
    #     """
    #     c, z_dim, y_dim, x_dim = input_image_np.shape

    #     all_patches = []
    #     all_slicers_info = []

    #     # --- 收集并预处理所有轴向的patches ---
    #     for axis_name in ['Z', 'Y', 'X']:
    #         slices = input_image_np[0, :, :, :]

    #         if axis_name == 'Z':
    #             spacings = np.array([original_spacing[1], original_spacing[0]])

    #         elif axis_name == 'Y':
    #             slices = np.transpose(slices, (1, 0, 2))
    #             spacings = np.array([original_spacing[2], original_spacing[0]])
    #         else:  # axis_name == 'X'
    #             slices = np.transpose(slices, (2, 0, 1))
    #             spacings = np.array([original_spacing[2], original_spacing[1]])

    #         indices = [slices.shape[0] // 2]
    #         if not np.any(slices[indices[0]]):
    #             indices.extend([slices.shape[0] // 4, slices.shape[0] * 3 // 4])

    #         for idx in indices:
    #             image_slice_np = slices[idx, :, :][None][None]

    #             # 1. 归一化
    #             image_slice_np = (image_slice_np - image_slice_np.mean()) / (image_slice_np.std().clip(1e-8))

    #             # 2. 重采样
    #             current_spacing = np.concatenate([np.array([1]), spacings])
    #             dst_shape = [1, int(round(image_slice_np.shape[2] * current_spacing[1] / target_spacing[1])),
    #                          int(round(image_slice_np.shape[3] * current_spacing[2] / target_spacing[2]))]

    #             image_resized = resample_torch_simple(
    #                 torch.from_numpy(image_slice_np).float().to(self.device),
    #                 dst_shape,
    #                 is_seg=False,
    #                 device=self.device,
    #                 mode='trilinear'
    #             )

    #             # 3. 填充和切片
    #             padded_image, slicer_revert_padding = pad_nd_image(image_resized, self.configuration_manager.patch_size, 'constant', {'value': 0}, True, None)
    #             slicers = self._internal_get_sliding_window_slicers(padded_image.shape[1:], self.configuration_manager.patch_size, self.tile_step_size)

    #             for i, sl in enumerate(slicers):
    #                 patch = padded_image[sl].cpu()
    #                 all_patches.append(patch)
    #                 all_slicers_info.append({
    #                     'axis': axis_name,
    #                     'idx': idx,
    #                     'slicer': sl,
    #                     'padded_shape': padded_image.shape,
    #                     'original_shape': image_slice_np.shape,
    #                     'revert_padding': slicer_revert_padding
    #                 })

    #     # --- 批量推理 ---
    #     if not all_patches:
    #         return np.zeros(input_image_np.shape[1:], dtype=np.uint8)

    #     full_batch = torch.stack(all_patches, dim=0).to(self.device)

    #     # 加载网络参数并设置评估模式
    #     if isinstance(self.network, OptimizedModule):
    #         self.network._orig_mod.load_state_dict(self.list_of_parameters[0])
    #     else:
    #         self.network.load_state_dict(self.list_of_parameters[0])

    #     self.network.to(self.device).eval()
    #     empty_cache(self.device)

    #     with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
    #         full_batch_predictions = self.network(full_batch)

    #     # --- 分配结果并后处理 ---
    #     aggregated_results = {}
    #     for info in all_slicers_info:
    #         key = (info['axis'], info['idx'])
    #         if key not in aggregated_results:
    #             num_classes = 2 # Assuming binary segmentation (vessel vs background)
    #             shape = info['padded_shape']
    #             aggregated_results[key] = {
    #                 'logits': torch.zeros((num_classes, *shape[1:]), dtype=torch.half, device=self.device),
    #                 'n_predictions': torch.zeros(shape[1:], dtype=torch.half, device=self.device),
    #                 'info': info
    #             }

    #     self.gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8, value_scaling_factor=10, device=self.device) if self.use_gaussian else 1
    #     for i, prediction in enumerate(full_batch_predictions):
    #         info = all_slicers_info[i]
    #         key = (info['axis'], info['idx'])
    #         if self.use_gaussian:
    #             prediction = prediction * self.gaussian

    #         sl = info['slicer']
    #         aggregated_results[key]['logits'][sl] += prediction
    #         aggregated_results[key]['n_predictions'][sl[1:]] += self.gaussian

    #     predictions = []
    #     for key, result in aggregated_results.items():
    #         info = result['info']
    #         logits = result['logits']
    #         n_predictions = result['n_predictions']

    #         logits /= n_predictions
    #         logits = logits[(slice(None), *info['revert_padding'][1:])]

    #         logits_resampled = resample_torch_simple(
    #             logits,
    #             info['original_shape'][1:],
    #             is_seg=False,
    #             device=self.device,
    #             mode='trilinear'
    #         )

    #         pred_array = logits_resampled.cpu().argmax(0).numpy()
    #         if np.any(pred_array > 0):
    #             predictions.append((info['axis'], 'center' if info['idx'] == input_image_np.shape[1] // 2 else 'sub', pred_array, info['idx']))

    #     # --- 融合所有预测结果并生成3D Bounding Box ---
    #     final_bbox_mask = None
    #     final_z_min, final_z_max = [], []
    #     final_y_min, final_y_max = [], []
    #     final_x_min, final_x_max = [], []

    #     if not predictions:
    #         return np.zeros(input_image_np.shape[1:], dtype=np.uint8)
    #     else:
    #         for axis, _, pred_slice, idx in predictions:
    #             if axis == 'Z':
    #                 z_min, z_max = idx, idx + 1
    #                 y_coords, x_coords = np.where(pred_slice.squeeze() > 0)
    #                 if len(y_coords) > 0:
    #                     y_min, y_max = y_coords.min(), y_coords.max() + 1
    #                     x_min, x_max = x_coords.min(), x_coords.max() + 1
    #                     final_y_min.append(y_min); final_y_max.append(y_max)
    #                     final_x_min.append(x_min); final_x_max.append(x_max)
    #             elif axis == 'Y':
    #                 y_min, y_max = idx, idx + 1
    #                 z_coords, x_coords = np.where(pred_slice.squeeze() > 0)
    #                 if len(z_coords) > 0:
    #                     z_min, z_max = z_coords.min(), z_coords.max() + 1
    #                     x_min, x_max = x_coords.min(), x_coords.max() + 1
    #                     final_z_min.append(z_min); final_z_max.append(z_max)
    #                     final_x_min.append(x_min); final_x_max.append(x_max)
    #             elif axis == 'X':
    #                 x_min, x_max = idx, idx + 1
    #                 z_coords, y_coords = np.where(pred_slice.squeeze() > 0)
    #                 if len(z_coords) > 0:
    #                     z_min, z_max = z_coords.min(), z_coords.max() + 1
    #                     y_min, y_max = y_coords.min(), y_coords.max() + 1
    #                     final_z_min.append(z_min); final_z_max.append(z_max)
    #                     final_y_min.append(y_min); final_y_max.append(y_max)

    #         z_min_final = int(np.mean(final_z_min)) if final_z_min else 0
    #         z_max_final = int(np.mean(final_z_max)) if final_z_max else z_dim
    #         y_min_final = int(np.mean(final_y_min)) if final_y_min else 0
    #         y_max_final = int(np.mean(final_y_max)) if final_y_max else y_dim
    #         x_min_final = int(np.mean(final_x_min)) if final_x_min else 0
    #         x_max_final = int(np.mean(final_x_max)) if final_x_max else x_dim

    #         final_bbox_mask = np.zeros(input_image_np.shape[1:], dtype=np.uint8)
    #         final_bbox_mask[z_min_final:z_max_final, y_min_final:y_max_final, x_min_final:x_max_final] = 1
    #         return final_bbox_mask

def get_bbox_coords(mask: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """
    Get the 3D bounding box coordinates from a 3D boolean mask.
    """
    if not np.any(mask):
        return None
    coords = np.where(mask)
    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1
    return z_min, z_max, y_min, y_max, x_min, x_max

def predict_on_gpu_efficient(gpu_id, files_queue, output_predict_folder, model_params):
    print(f"Process {os.getpid()} is using GPU {gpu_id}...")

    # 初始化预测器
    predictor = nnXNetPredictor(
        tile_step_size=0.5,
        use_mirroring=False,
        use_gaussian=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', gpu_id),
        allow_tqdm=False
    )

    # 从训练好的模型文件夹初始化网络和加载权重
    predictor.initialize_from_trained_model_folder(
        join(nnXNet_results, model_params['dataset_name_and_id'], model_params['trainer_name'] + '__' + model_params['plans_name'] + '__' + model_params['configuration_name']),
        use_folds=model_params['folds_to_use'],
        checkpoint_name=model_params['checkpoint_name'],
    )

    predictor.initialize_network_and_gaussian()
    
    while True:
        file_name = files_queue.get()
        if file_name is None:
            break

        input_file_path = join(input_folder, file_name)
        print(f"GPU {gpu_id} started processing file: {file_name}")

        input_img_np, input_props = SimpleITKIO().read_images([input_file_path])
        original_direction = input_props['sitk_stuff']['direction']
        original_origin = input_props['sitk_stuff']['origin']
        original_spacing = input_props['sitk_stuff']['spacing']

        # 目标间距
        target_spacing = np.array([1, 0.6, 0.4492])

        # 使用封装好的方法进行预测
        final_bbox_mask = predictor.predict_from_multi_axial_slices(input_img_np, original_spacing, target_spacing)

        # Save the final bbox mask
        output_img_sitk = sitk.GetImageFromArray(final_bbox_mask)
        output_img_sitk.SetDirection(original_direction)
        output_img_sitk.SetOrigin(original_origin)
        output_img_sitk.SetSpacing(original_spacing)

        output_file_name_base = file_name.replace('_0000', '')
        output_file_name = join(output_predict_folder, output_file_name_base)
        sitk.WriteImage(output_img_sitk, output_file_name, useCompression=True)
        print(f"GPU {gpu_id} successfully saved final bbox prediction for: {output_file_name}")

    print(f"Process {os.getpid()} finished.")

# --- 主执行块 (Main execution block) ---
if __name__ == '__main__':
    # --- 定义路径和参数 (Paths and Parameters) ---
    nnXNet_results = "/yinghepool/shipengcheng/Dataset/nnXNet/nnXNet_results"
    dataset_name_and_id = 'Dataset180_2D_vessel_box_seg'
    trainer_name = 'nnXNetTrainer'
    plans_name = 'nnXNetPlans'
    configuration_name = '2d'
    input_folder = '/yinghepool/shipengcheng/Dataset/nnXNet/nnXNet_raw/Dataset630_vessel_anatomy_aneurysm_26classes_2279/imagesTr'
    output_predict_folder = '/yinghepool/shipengcheng/Dataset/nnXNet/nnXNet_raw/Dataset630_vessel_anatomy_aneurysm_26classes_2279/pred_vessel_3D_box_fast_0913_2'
    checkpoint_name = 'checkpoint_final.pth'
    folds_to_use = (0,)

    # 定制每个GPU的进程数量 (Customize the number of processes per GPU)
    gpu_processes_config = {
        0: 1,  # 使用GPU 0运行1个进程
    }

    # Create the output folder if it doesn't exist
    os.makedirs(output_predict_folder, exist_ok=True)

    files_to_process = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]
    files_queue = Queue()

    for f in files_to_process:
        files_queue.put(f)

    total_processes = sum(gpu_processes_config.values())
    print(f"Found {len(files_to_process)} files. Using {total_processes} processes across {len(gpu_processes_config)} GPUs.")

    processes = []
    for gpu_id, num_processes in gpu_processes_config.items():
        for _ in range(num_processes):
            p = Process(target=predict_on_gpu_efficient, args=(
                gpu_id,
                files_queue,
                output_predict_folder,
                {'dataset_name_and_id': dataset_name_and_id,
                 'trainer_name': trainer_name,
                 'plans_name': plans_name,
                 'configuration_name': configuration_name,
                 'folds_to_use': folds_to_use,
                 'checkpoint_name': checkpoint_name}
            ))
            p.start()
            processes.append(p)

    for _ in range(total_processes):
        files_queue.put(None)

    for p in processes:
        p.join()

    print("\n--- All files processed. ---")