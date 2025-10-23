# [RSNA2025 Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) - 2nd Place Solution

**Fast 2D tri-axial ROI extraction + 3D Multi-Task Segmentation and Classification**
The solution write-up is available at: [RSNA2025 2nd-place-solution](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)

# Training Instructions

## Overview

Our approach focused on simplicity and generality to handle the diverse data in this classification-focused task.

**Key Elements:**

- **Stage 1**: Fast 2D tri-axial ROI extraction using an nnU-Net 2D segmentation model to crop binary vascular regions efficiently, validated on all training cases locally.

- **Stage 2**: 3D multi-task learning (segmentation of vessels and aneurysms + classification) based on nnU-Net, with enhancements including:
  - Cross-attention pooling
  - Modality 4-class heads
  - Targeted oversampling for rare classes
  - All data resized to uniform 224×224×224
  - Heavy TTA (8×) including left-right flips with label swapping

# Hardware Used

- CPU: Intel(R) Xeon(R) Platinum 8468 @2.10GHz
- Memory: 512 GiB RAM
- GPU: 1 × NVIDIA A100 80GB
- Operating System: Ubuntu 22.04.4 LTS

# Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/PengchengShi1220/RSNA2025_Intracranial-Aneurysm-Detection
```
2. Install requirements:
```bash
pip install -r RSNA2025_Intracranial-Aneurysm-Detection/requirements.txt
```
3. Install nnXNet package:
```bash
pip install -e RSNA2025_Intracranial-Aneurysm-Detection/nnXNet
```

# Processing DICOM to NIfTI
Use [process_RSNA2025_all_data.py](https://github.com/PengchengShi1220/RSNA2025_Intracranial-Aneurysm-Detection/blob/master/process_RSNA2025_all_data.py) to convert all DICOM images into individual .nii.gz files, while transforming the coordinates from `train_localizers.csv` into 3D NIfTI format with binary mask images centered around the 3D bounding boxes.

DICOM to NIfTI conversion can be performed using `dicom2nifti`, with the `reorient_nii` function applied to standardize orientation to "LPS". This process is implemented according to [nifti_by_dicom2nifti.py](https://github.com/PengchengShi1220/RSNA2025_Intracranial-Aneurysm-Detection/blob/master/nifti_by_dicom2nifti.py).

# Detailed Training Procedure

## Stage 1: 2D Segmentation Model
The first stage uses nnUNetv2 for 2D vessel segmentation:

- Stage 1 training data ([504 sample cases](https://www.kaggle.com/datasets/pengchengshi/dataset180-2d-vessel-box-seg)), nnU-Net format:

```bash
# Dataset planning and preprocessing
nnUNetv2_plan_and_preprocess -d 180 -c "2d" --verify_dataset_integrity

# Model training
nnUNetv2_train Dataset180_2D_vessel_box_seg 2d 0 -tr nnUNetTrainer --c
```

## Stage 2: 3D Multi-task Learning
The second stage employs nnXNet for 3D vessel anatomy and aneurysm segmentation with 26 classes:

### Initial Training (1000 epochs)
```bash
nnXNet_train Dataset660_vessel_anatomy_aneurysm_26classes_resize224_4661 3d_fullres 0 \
  -tr nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality_CE_DC_AWDC_onlyMirror01 \
  -p nnXNetResEncUNetM_two_seg_with_cls_ps_224_224_224_Plans
```

### Fine-tuning Phase 1 (100 epochs)
```bash
nnXNet_train Dataset660_vessel_anatomy_aneurysm_26classes_resize224_4661 3d_fullres 0 \
  -tr nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality_CE_DC_AWDC_onlyMirror01_lr4e3_100epochs \
  -p nnXNetResEncUNetM_two_seg_with_cls_ps_224_224_224_Plans
```

### Fine-tuning Phase 2 (250 epochs)
```bash
nnXNet_train Dataset660_vessel_anatomy_aneurysm_26classes_resize224_4661 3d_fullres 1 \
  -tr nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality_CE_DC_AWDC_onlyMirror01_250epochs \
  -p nnXNetResEncUNetM_two_seg_with_cls_ps_224_224_224_Plans
```

**Note:** Stage 2 training uses batch size = 2 with approximately 53GB GPU memory consumption.

# Code Resources

**Inference Notebook:**  
- [bravecowcow-2nd-place-inference-demo.ipynb](https://www.kaggle.com/code/pengchengshi/bravecowcow-2nd-place-inference-demo)
- [bravecowcow-2nd-place-inference-final-submission.ipynb](https://www.kaggle.com/code/pengchengshi/bravecowcow-2nd-place-inference)

**For upcoming versions and paper, please follow:**  
[nnX-Net: An Extensible Multi-task Learning Framework for Medical Imaging](https://github.com/yinghemedical/nnXNet)

## Acknowledgements
Thanks to Medical Image Insights and UZH for compute support, Bjoern Menze and the Helmut Horten Foundation for funding support. We are grateful to RSNA/Kaggle hosts, [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/master) devs, and forum contributors.
