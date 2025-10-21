# [RSNA2025 Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) - 2nd Place Solution

**Fast 2D tri-axial ROI extraction + 3D Multi-Task Segmentation and Classification**

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

## Code Resources

**Inference Notebook:**  
[bravecowcow-2nd-place-inference.ipynb](https://github.com/PengchengShi1220/RSNA2025_Intracranial-Aneurysm-Detection/blob/master/bravecowcow-2nd-place-inference.ipynb)

**Current Implementation:**
- Stage 2 is based on nnXNet 0.5.0 version
- Repository: [nnXNet 0.5.0](https://github.com/PengchengShi1220/RSNA2025_Intracranial-Aneurysm-Detection/tree/master/nnXNet)

**Model Checkpoint:**
- Stage 1 model checkpoint: [rsna2025-stage1-model](https://www.kaggle.com/models/pengchengshi/dataset180_2d_vessel_box_seg_stable)
- Stage 2 model checkpoint: [rsna2025-stage2-models](https://www.kaggle.com/models/pengchengshi/rsna2025-stage2-models)

**For upcoming versions and paper, please follow:**  
[nnX-Net: An Extensible Multi-task Learning Framework for Medical Imaging](https://github.com/yinghemedical/nnXNet)

## Acknowledgements
Thanks to Medical Image Insights and UZH for compute support, Bjoern Menze and the Helmut Horten Foundation for funding support. We are grateful to RSNA/Kaggle hosts, [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/master) devs, and forum contributors.
