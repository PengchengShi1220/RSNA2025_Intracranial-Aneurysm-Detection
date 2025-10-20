from typing import Tuple, Union, List, Type
import torch
import torch.nn as nn
import os
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BasicBlockD
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from timm.layers import ClassifierHead
    
class ResEncoderUNet_with_cls(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cls_head_num_classes_list: List[int] = [1, 4], 
        cls_drop_out_list: List[int] = [0.5, 0.5],
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
        self.out_channels = out_channels
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
        
        transpconv_op = get_matching_convtransp(conv_op=conv_op)
        # Decoder Blocks
        self.transpconvs, self.decoder_blocks, self.seg_layers = self.build_decoder_blocks(transpconv_op, features_per_stage, kernel_sizes, strides, n_conv_per_stage_decoder, conv_op, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first)

        query_num = 1
        self.cls_head_list = nn.ModuleList()
        self.cls_drop_out_list = cls_drop_out_list #[0.5, 0.5]
        for i in range(len(cls_head_num_classes_list)):
            cls_head_num_classes = cls_head_num_classes_list[i]
            cls_drop_out = self.cls_drop_out_list[i]
            self.cls_head_list.append(ClassificationHead(features_per_stage[-1], query_num, cls_head_num_classes, dropout=cls_drop_out, use_cross_attention=False, num_heads=4))

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

    def build_decoder_blocks(self, transpconv_op, features_per_stage, kernel_sizes, strides, n_conv_per_stage_decoder, conv_op, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first):
        transpconvs = nn.ModuleList()
        decoder_blocks = nn.ModuleList()
        seg_layers = nn.ModuleList()

        for i in range(len(features_per_stage) - 1, 0, -1):
            transpconvs.append(
                    transpconv_op(
                        features_per_stage[i], features_per_stage[i - 1], strides[i], strides[i],
                        bias=conv_bias
                    )
                )
            decoder_blocks.append(
                    StackedConvBlocks(
                        n_conv_per_stage_decoder[i - 1], conv_op, 2 * features_per_stage[i - 1], features_per_stage[i - 1],
                        kernel_sizes[i], 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, nonlin_first
                    )
                )
            seg_layers.append(conv_op(features_per_stage[i - 1], self.out_channels, kernel_size=1, stride=1, padding=0, bias=True))

        return transpconvs, decoder_blocks, seg_layers

    def forward(self, input_image, only_forward_cls=False):

        conv_enc_outputs = [self.conv_encoder_blocks[0](input_image)]
        for i in range(1, len(self.conv_encoder_blocks)):
            conv_enc_outputs.append(self.conv_encoder_blocks[i](conv_enc_outputs[-1]))

        lres_input = conv_enc_outputs[-1]

        # cls:
        # lres_input_avg = lres_input.mean(dim=[2, 3, 4])
        cls_pred_list = []
        for cls_head in self.cls_head_list:
            cls_pred_list.append(cls_head(lres_input)) #lres_input_avg))

        if only_forward_cls:
            return cls_pred_list
        else:
            seg_outputs = []
            for s in range(len(self.decoder_blocks)):
                x = self.transpconvs[s](lres_input)
                x = torch.cat((x, conv_enc_outputs[-(s+2)]), 1)
                x = self.decoder_blocks[s](x)
                if self.deep_supervision:
                    seg_outputs.append(self.seg_layers[s](x))
                elif s == (len(self.decoder_blocks) - 1):
                    seg_outputs.append(self.seg_layers[-1](x))
                lres_input = x

            if self.deep_supervision:
                r = seg_outputs[::-1]
            else:
                r = seg_outputs[-1]

            return r, cls_pred_list

class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes=14, num_heads=4, dropout=0.1):
        super(CrossAttentionPooling, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # 可学习的查询向量，形状为 [1, embed_dim]（单个查询）
        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))
        
        # Cross Attention 层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # 使用 (seq_len, batch, embed_dim) 格式
        )
        
        # LayerNorm 和 Dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 分类器：将 [1, D] 映射到 [1, num_classes]
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: 图像特征 [B, D, H, W, L] 或 [B, D, H*W*L]
        
        Returns:
            分类logits [B, num_classes]
        """
        # print("x.shape: ", x.shape)
        batch_size = x.shape[0]
        
        # 处理输入特征
        if x.dim() == 5:  # [B, D, H, W, L]
            # 展平空间维度
            x = x.flatten(2)  # [B, D, H*W*L]
        
        # 调整维度: [B, D, L] -> [L, B, D] (seq_len, batch, embed_dim)
        x = x.permute(2, 0, 1)  # [H*W*L, B, D]
        
        # 扩展查询向量: [query_num, embed_dim] -> [query_num, B, D]
        query = self.class_query.unsqueeze(1).repeat(1, batch_size, 1)  # [query_num, B, D]
        
        # Cross Attention
        attended, attention_weights = self.cross_attention(
            query=query,  # [query_num, B, D] - 单个查询
            key=x,        # [L, B, D]  
            value=x,      # [L, B, D]
        )
        
        # attended 形状: [query_num, B, D]
        attended = self.norm(attended)
        attended = self.dropout(attended)
        
        # 调整维度: [query_num, B, D] -> [B, D]
        attended_avg = attended.mean(dim=[0]).squeeze(0)  # [B, D]
        # print("attended_avg.shape: ", attended_avg.shape)
        
        # 应用分类器：将 [B, D] 映射到 [B, num_classes]
        logits = self.classifier(attended_avg)  # [B, num_classes]
        
        return logits


class ClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim,
        query_num,
        num_classes,
        dropout=0.5,
        use_cross_attention=True,
        num_heads=4
    ):
        """
        Args:
            embed_dim (int): 嵌入维度
            num_classes (int): 类别数量
            dropout (float): dropout率
            use_cross_attention (bool): 是否使用cross attention pooling
            num_heads (int): attention头数
        """
        super(ClassificationHead, self).__init__()
        
        if use_cross_attention:
            self.pooling = CrossAttentionPooling(
                embed_dim=embed_dim,
                query_num=query_num,
                num_classes=num_classes,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),  # 全局平均池化
                nn.Flatten(1),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes)
            )
    
    def forward(self, x):
        return self.pooling(x)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = torch.randn(size=(2, 1, 256, 256, 128)).to(device)
    
    net = ResEncoderUNet_with_cls(
        in_channels=1,
        out_channels=6,
        pos_weights_list=[1],
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
        transpconv_op=nn.ConvTranspose3d,
        deep_supervision=True
    ).to(device)
    out = net(inputs)
    for i in range(len(out)):
        print(out[i].shape)
