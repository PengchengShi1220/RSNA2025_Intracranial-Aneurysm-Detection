from nnxnet.training.nnXNetTrainer.nnXNetTrainer_ResEncoderUNet_with_cls import nnXNetTrainer_ResEncoderUNet_with_cls

class nnXNetTrainer_ResEncoderUNet_with_cls_NoMirroring(nnXNetTrainer_ResEncoderUNet_with_cls):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
        