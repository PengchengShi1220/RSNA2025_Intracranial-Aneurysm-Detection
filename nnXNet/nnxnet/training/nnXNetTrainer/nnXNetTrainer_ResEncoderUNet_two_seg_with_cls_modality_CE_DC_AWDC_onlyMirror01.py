from nnxnet.training.nnXNetTrainer.nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality_CE_DC_AWDC import nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality_CE_DC_AWDC

class nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality_CE_DC_AWDC_onlyMirror01(nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality_CE_DC_AWDC):
    """
    Only mirrors along spatial axes 0 and 1 for 3D and 0 for 2D
    """
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 2:
            mirror_axes = (0, )
        else:
            mirror_axes = (0, 1)
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
    