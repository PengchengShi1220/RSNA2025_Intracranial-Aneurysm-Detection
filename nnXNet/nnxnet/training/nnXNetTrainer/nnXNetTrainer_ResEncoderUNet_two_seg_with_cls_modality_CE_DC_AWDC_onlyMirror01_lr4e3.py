import torch

from nnxnet.training.nnXNetTrainer.nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality_CE_DC_AWDC_onlyMirror01 import nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality_CE_DC_AWDC_onlyMirror01

class nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality_CE_DC_AWDC_onlyMirror01_lr4e3(nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality_CE_DC_AWDC_onlyMirror01):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 4e-3