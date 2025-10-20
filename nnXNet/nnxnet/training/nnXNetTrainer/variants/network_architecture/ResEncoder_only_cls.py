from typing import Tuple, Union, List, Type
import torch
import torch.nn as nn
import os
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BasicBlockD
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from timm.layers import ClassifierHead

# --------------------------------------------------------------------------------
ACT_TYPE = {"relu": nn.ReLU, "gelu": nn.GELU}

class PositionalEncoding(nn.Module):
    def __init__(self, coord_dim=6, feature_dim=1024, hidden_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, coords):
        pos_embedding = self.mlp(coords)
        return pos_embedding

class AttnPoolingWithPos(nn.Module):
    def __init__(self, input_dim, output_dim, n_queries=256, n_head=8, mlp_depth=2, act_type='relu'):
        super().__init__()
        self.n_queries = n_queries
        self.query = nn.Parameter(torch.randn(n_queries, input_dim))
        
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=n_head, 
            batch_first=True
        )

        modules = [nn.Linear(input_dim, output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(output_dim, output_dim))

        self.connector = nn.Sequential(*modules)
        
    def forward(self, x):
        batch_size, num_patches, d_model = x.shape
        
        q = self.query.unsqueeze(0).repeat(batch_size, 1, 1)
        # print("q shape:", q.shape)  # Debugging line
        # print("x shape:", x.shape)  # Debugging line
        attn_output, _ = self.attn(q, x, x)
        
        x = self.connector(attn_output)

        return x
# --------------------------------------------------------------------------------

class ResEncoder_only_cls(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cls_head_num_classes_list: List[int] = [1, 4], 
        n_stages: int = 6,
        features_per_stage: List[int] = [16, 32, 64, 128, 256, 320],
        kernel_sizes: List[Union[Tuple[int, int, int], int]] = [(3, 3, 1), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
        strides: List[Union[Tuple[int, int, int], int]] = [(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        dropout_rate: float = 0.0,
        deep_supervision: bool = True,
        norm_op: nn.Module = nn.InstanceNorm3d,
        norm_op_kwargs: dict = {"eps": 1e-05, "affine": True},
        conv_op: nn.Module = nn.Conv3d,
        conv_bias: bool = True,
        nonlin: nn.Module = nn.LeakyReLU,
        nonlin_kwargs: dict = {"inplace": True},
        dropout_op: Union[None, Type[nn.Module]] = None,
        dropout_op_kwargs: dict = None,
        n_blocks_per_stage: List[int] = [2, 2, 2, 2, 2, 2],
        n_conv_per_stage_decoder: List[int] = [2, 2, 2, 2, 2],
        nonlin_first: bool = False,
    ) -> None:
        super().__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.in_channels = in_channels
        self.deep_supervision = deep_supervision
        num_stages = len(features_per_stage)
        self.num_stages = num_stages

        kernel_sizes = kernel_sizes or [(3, 3, 3) for _ in range(num_stages)]
        strides = strides or [(2, 2, 2) for _ in range(num_stages)]
        norm_op_kwargs = norm_op_kwargs or {"eps": 1e-05, "affine": True}
        nonlin_kwargs = nonlin_kwargs or {"inplace": True}
        n_blocks_per_stage = n_blocks_per_stage or [2 for _ in range(num_stages)]
        n_conv_per_stage_decoder = n_conv_per_stage_decoder or [2 for _ in range(num_stages - 1)]

        # Convolutional Encoder Blocks (U-Net style)
        self.conv_encoder_blocks = self.build_encoder_block(n_blocks_per_stage, features_per_stage, kernel_sizes, strides, in_channels, conv_op, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, BasicBlockD)
        
        # Patch-level classification heads
        self.cls_head_list = nn.ModuleList()
        for cls_head_num_classes in cls_head_num_classes_list:
            self.cls_head_list.append(ClassificationHead(features_per_stage[-1], cls_head_num_classes))

        # Positional and Attention Pooling
        self.pos_encoder = PositionalEncoding(coord_dim=6, feature_dim=features_per_stage[-1])
        
        # 使用传入的attn_pooling_kwargs来配置AttnPoolingWithPos
        self.attn_pooling = AttnPoolingWithPos(
            input_dim=features_per_stage[-1], 
            output_dim=features_per_stage[-1],
            n_queries=256,
            n_head=8,
            mlp_depth=2,
            act_type='relu'
        )

        # Full-volume classification heads (for aggregated features)
        self.agg_cls_head_list = nn.ModuleList()
        for cls_head_num_classes in cls_head_num_classes_list:
            self.agg_cls_head_list.append(ClassificationHead(features_per_stage[-1], cls_head_num_classes))

    def build_encoder_block(self, n_blocks_per_stage, features_per_stage, kernel_sizes, strides, in_channels, conv_op, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block_type):
        blocks = nn.ModuleList()
        blocks.append(
            StackedResidualBlocks(
                n_blocks_per_stage[0], conv_op, in_channels, features_per_stage[0], kernel_sizes[0], strides[0],
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                block=block_type
            )
        )
        for i in range(1, len(n_blocks_per_stage)):
            blocks.append(
                StackedResidualBlocks(
                    n_blocks_per_stage[i], conv_op, features_per_stage[i - 1], features_per_stage[i], kernel_sizes[i],
                    strides[i], conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    block=block_type
                )
            )

        return blocks

    def forward(self, input_image: torch.Tensor, coords: torch.Tensor):
        """
        Forward pass for a batch of patches.
        Extracts features and performs patch-level classification.
        
        Args:
            input_image (torch.Tensor): Tensor of shape (batch_size, 1, D, H, W).
            coords (torch.Tensor): Tensor of shape (batch_size, 6).
        """
        # 1. 卷积编码器
        conv_enc_outputs = [self.conv_encoder_blocks[0](input_image)]
        for i in range(1, len(self.conv_encoder_blocks)):
            conv_enc_outputs.append(self.conv_encoder_blocks[i](conv_enc_outputs[-1]))

        lres_input = conv_enc_outputs[-1] # shape (batch_size, feature_dim, d, h, w)
        
        # 2. 全局平均池化得到 patch 级别的特征
        patch_features = lres_input.mean(dim=[2, 3, 4]) # shape (batch_size, feature_dim)

        # # 3. Positional Encoding
        # # coords 的 shape 应该是 (batch_size, 6)，因此可以直接传入
        # pos_embeddings = self.pos_encoder(coords) # shape (batch_size, feature_dim)

        # 4. Additive fusion of features and positional info
        fused_features = patch_features #+ pos_embeddings

        # 5. Patch-level classification predictions
        cls_pred_list = [cls_head(fused_features) for cls_head in self.cls_head_list]

        # 返回融合后的特征和分类预测
        return cls_pred_list, fused_features

    def attn_pooling_cls(self, patch_features: torch.Tensor):
        """
        Performs full-volume classification by aggregating patch features.
        This method is designed to be called with ALL patches from a single case.
        
        Args:
            patch_features (torch.Tensor): Tensor of shape (num_patches, feature_dim)
                                            or (1, num_patches, feature_dim)
        Returns:
            List[torch.Tensor]: A list of aggregated predictions for each classification task.
        """
        # Ensure the input is 2D or 3D tensor
        if patch_features.dim() == 2:
            # Add a batch dimension if it's not present
            patch_features = patch_features.unsqueeze(0) # shape (1, num_patches, feature_dim)
        elif patch_features.dim() != 3 or patch_features.shape[0] != 1:
            raise ValueError("Input to forward_agg must be a 2D tensor (num_patches, feature_dim) or a 3D tensor (1, num_patches, feature_dim)")

        # print("patch_features shape:", patch_features.shape)  # Debugging line
        # 1. Cross-attention pooling
        aggregated_features = self.attn_pooling(patch_features) # shape (1, n_queries, feature_dim)
        # print("aggregated_features shape:", aggregated_features.shape)  # Debugging line

        # 2. Global feature extraction (e.g., mean pooling over queries)
        final_global_feature = aggregated_features.mean(dim=1) # shape (1, feature_dim)

        # 3. Full-volume classification
        agg_pred_list = [agg_head(final_global_feature) for agg_head in self.agg_cls_head_list]

        return agg_pred_list

class ClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_classes,
        dropout=0.0,
    ):
        super(ClassificationHead, self).__init__()
        self.fc = ClassifierHead(embed_dim, num_classes, "", dropout)

    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模拟一个批次（batch_size=2）的patches和它们的坐标
    inputs = torch.randn(size=(2, 1, 96, 192, 128)).to(device)
    # 模拟每个 patch 的坐标，形状为 (batch_size, 6)
    patch_coords = torch.rand(size=(2, 6)).to(device)
    
    # 实例化网络
    net = ResEncoder_only_cls(
        in_channels=1,
        cls_head_num_classes_list=[1, 4], 
        n_stages=6,
        features_per_stage=[16, 32, 64, 128, 256, 320],
        kernel_sizes=[(3, 3, 1), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
        strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        conv_op=nn.Conv3d,
        conv_bias=True,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        n_blocks_per_stage=[2, 2, 2, 2, 2, 2],
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        nonlin_first=True,
        attn_pooling_kwargs={"n_queries": 256, "n_head": 8, "mlp_depth": 2, "act_type": "gelu"}
    ).to(device)

    # 1. Patch级别的处理 (在训练循环的内部)
    # 此时调用 forward 时，必须同时传入 patches 和对应的 coords
    cls_pred_list, fused_patch_features = net(inputs, patch_coords)
    
    print("Patch-level classification outputs:")
    for i, pred in enumerate(cls_pred_list):
        print(f"Task {i} prediction shape: {pred.shape}")
    print("\nFused patch features shape for aggregation:", fused_patch_features.shape)

    # 2. 全图级别的聚合和分类 (在训练循环的外部, 针对一个完整病例)
    # 模拟一个病例的所有patches的融合特征
    num_patches_per_case = 80
    # 在这里使用融合后的特征进行聚合
    all_case_features = torch.randn(num_patches_per_case, 320).to(device)
    # 注意：forward_agg 现在不再需要 coords 参数
    
    # 调用新的 forward_agg 方法
    full_volume_preds = net.attn_pooling_cls(all_case_features)
    
    print("\nFull-volume classification outputs:")
    for i, pred in enumerate(full_volume_preds):
        print(f"Aggregated Task {i} prediction shape: {pred.shape}")