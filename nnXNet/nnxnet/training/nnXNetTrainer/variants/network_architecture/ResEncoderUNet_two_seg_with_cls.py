from typing import Tuple, Union, List, Type
import torch
import torch.nn as nn
import os
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BasicBlockD
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

# ----------------- 自定义的 CrossAttentionPooling 以支持多查询和拼接 -----------------
class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes, num_heads=4, dropout=0.0):
        super(CrossAttentionPooling, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num
        
        # 可学习的查询向量，形状为 [query_num, embed_dim] 
        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))
        
        # Cross Attention 层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        
        # LayerNorm 和 Dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 分类器：将 [query_num * D] 映射到 [num_classes]
        # 注意这里是 query_num * embed_dim
        self.classifier = nn.Linear(query_num * embed_dim, num_classes) 
        
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
        batch_size = x.shape[0]
        
        # 处理输入特征
        if x.dim() == 5:  # [B, D, H, W, L]
            x = x.flatten(2)  # [B, D, H*W*L]
        
        # 调整维度: [B, D, L] -> [L, B, D] (seq_len, batch, embed_dim)
        x = x.permute(2, 0, 1)  # [H*W*L, B, D]
        
        # 扩展查询向量: [query_num, embed_dim] -> [query_num, B, D]
        query = self.class_query.unsqueeze(1).repeat(1, batch_size, 1)  # [query_num, B, D]
        
        # Cross Attention
        attended, attention_weights = self.cross_attention(
            query=query, 
            key=x,        
            value=x,      
        )
        
        # attended 形状: [query_num, B, D]
        attended = self.norm(attended)
        attended = self.dropout(attended)
        
        # **************************** 关键修改 ****************************
        # 调整维度: [query_num, B, D] -> [B, query_num, D]
        attended_permuted = attended.permute(1, 0, 2)
        
        # 展平查询维度和特征维度: [B, query_num, D] -> [B, query_num * D]
        # 这样能保留不同查询学到的信息
        attended_flatten = attended_permuted.flatten(1) 
        
        # 应用分类器：将 [B, query_num * D] 映射到 [B, num_classes]
        logits = self.classifier(attended_flatten)  # [B, num_classes]
        
        return logits


class ClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim,
        query_num,
        num_classes,
        dropout=0.0,
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
# ----------------- UNet 模型 -----------------

class ResEncoderUNet_two_seg_with_cls(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_1: int,
        out_channels_2: int,
        cls_head_num_classes_list: List[int] = [1, 13],
        cls_drop_out_list: List[int] = [0.0, 0.0], 
        cls_query_num_list: List[int] = [2, 16],
        use_cross_attention: bool = True,
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
        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2
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
        self.transpconvs, self.transpconvs_last_two_1, self.transpconvs_last_two_2, self.decoder_blocks, self.decoder_blocks_last_two_1, self.decoder_blocks_last_two_2, self.seg_layers_1, self.seg_layers_2 = self.build_decoder_blocks(transpconv_op, features_per_stage, kernel_sizes, strides, n_conv_per_stage_decoder, conv_op, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first)
        
        self.cls_head_list = nn.ModuleList()
        self.cls_drop_out_list = cls_drop_out_list
        self.cls_query_num_list = cls_query_num_list # 新增
        
        for i in range(len(cls_head_num_classes_list)):
            cls_head_num_classes = cls_head_num_classes_list[i]
            cls_drop_out = self.cls_drop_out_list[i]
            cls_query_num = self.cls_query_num_list[i] # 获取查询数量
            
            # 使用修改后的 ClassificationHead
            self.cls_head_list.append(ClassificationHead(
                embed_dim=features_per_stage[-1], 
                query_num=cls_query_num, 
                num_classes=cls_head_num_classes, 
                dropout=cls_drop_out, 
                use_cross_attention=use_cross_attention, 
                num_heads=4
            ))

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
        transpconvs_last_two_1 = nn.ModuleList()
        transpconvs_last_two_2 = nn.ModuleList()
        decoder_blocks = nn.ModuleList()
        decoder_blocks_last_two_1 = nn.ModuleList()
        decoder_blocks_last_two_2= nn.ModuleList()
        seg_layers_1 = nn.ModuleList()
        seg_layers_2 = nn.ModuleList()

        for i in range(len(features_per_stage) - 1, 0, -1):
            
            if i >= 5:
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
            else:
                transpconvs_last_two_1.append(
                    transpconv_op(
                        features_per_stage[i], features_per_stage[i - 1], strides[i], strides[i],
                        bias=conv_bias
                    )
                )
                transpconvs_last_two_2.append(
                    transpconv_op(
                        features_per_stage[i], features_per_stage[i - 1], strides[i], strides[i],
                        bias=conv_bias
                    )
                )
                decoder_blocks_last_two_1.append(
                        StackedConvBlocks(
                            n_conv_per_stage_decoder[i - 1], conv_op, 2 * features_per_stage[i - 1], features_per_stage[i - 1],
                            kernel_sizes[i], 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                            nonlin, nonlin_kwargs, nonlin_first
                        )
                    )
                decoder_blocks_last_two_2.append(
                        StackedConvBlocks(
                            n_conv_per_stage_decoder[i - 1], conv_op, 2 * features_per_stage[i - 1], features_per_stage[i - 1],
                            kernel_sizes[i], 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                            nonlin, nonlin_kwargs, nonlin_first
                        )
                    )
            seg_layers_1.append(conv_op(features_per_stage[i - 1], self.out_channels_1, kernel_size=1, stride=1, padding=0, bias=True))
            seg_layers_2.append(conv_op(features_per_stage[i - 1], self.out_channels_2, kernel_size=1, stride=1, padding=0, bias=True))

        return transpconvs, transpconvs_last_two_1, transpconvs_last_two_2, decoder_blocks, decoder_blocks_last_two_1, decoder_blocks_last_two_2, seg_layers_1, seg_layers_2

    def forward(self, input_image, only_forward_cls=False):

        conv_enc_outputs = [self.conv_encoder_blocks[0](input_image)]
        for i in range(1, len(self.conv_encoder_blocks)):
            conv_enc_outputs.append(self.conv_encoder_blocks[i](conv_enc_outputs[-1]))

        lres_input = conv_enc_outputs[-1]

        # cls:
        cls_pred_list = []
        for cls_head in self.cls_head_list:
            cls_pred_list.append(cls_head(lres_input)) 

        if only_forward_cls:
            return cls_pred_list
        else:
            seg_outputs_1 = []
            seg_outputs_2 = []
            for s in range(len(self.decoder_blocks + self.decoder_blocks_last_two_1)):
                
                if s <= (len(self.decoder_blocks + self.decoder_blocks_last_two_1) - 5):
                    x = self.transpconvs[s](lres_input)
                    x = torch.cat((x, conv_enc_outputs[-(s+2)]), 1)
                    x = self.decoder_blocks[s](x)

                    if self.deep_supervision:
                        seg_outputs_1.append(self.seg_layers_1[s](x))
                        seg_outputs_2.append(self.seg_layers_2[s](x))
                    lres_input = x

                else:
                    if s == (len(self.decoder_blocks + self.decoder_blocks_last_two_1) - 4):
                        lres_input_1 = lres_input
                        lres_input_2 = lres_input

                    x_1 = self.transpconvs_last_two_1[s-len(self.decoder_blocks)](lres_input_1)
                    x_2 = self.transpconvs_last_two_2[s-len(self.decoder_blocks)](lres_input_2)
                    x_1 = torch.cat((x_1, conv_enc_outputs[-(s+2)]), 1)
                    x_2 = torch.cat((x_2, conv_enc_outputs[-(s+2)]), 1)
                    x_1 = self.decoder_blocks_last_two_1[s-len(self.decoder_blocks)](x_1)
                    x_2 = self.decoder_blocks_last_two_2[s-len(self.decoder_blocks)](x_2)

                    if self.deep_supervision:
                        seg_outputs_1.append(self.seg_layers_1[s](x_1))
                        seg_outputs_2.append(self.seg_layers_2[s](x_2))
                    elif s == (len(self.decoder_blocks + self.decoder_blocks_last_two_1) - 1):
                        seg_outputs_1.append(self.seg_layers_1[-1](x_1))
                        seg_outputs_2.append(self.seg_layers_2[-1](x_2))
                    lres_input_1 = x_1
                    lres_input_2 = x_2

            if self.deep_supervision:
                r_1 = seg_outputs_1[::-1]
                r_2 = seg_outputs_2[::-1]
            else:
                r_1 = seg_outputs_1[-1]
                r_2 = seg_outputs_2[-1]

            return r_1, r_2, cls_pred_list

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = torch.randn(size=(2, 1, 256, 256, 128)).to(device)
    
    # 调整参数：query_num=4，dropout=0.5
    net = ResEncoderUNet_two_seg_with_cls(
        in_channels=1,
        out_channels_1=14,
        out_channels_2=15,
        cls_head_num_classes_list=[1, 13],
        cls_drop_out_list=[0.0, 0.0],
        cls_query_num_list=[2, 16],
        use_cross_attention=True,
        # 原始参数
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
        deep_supervision=True
    ).to(device)
    
    # 辅助测试，确保 forward 还能工作
    with torch.no_grad():
        out = net(inputs)
        print("Test Output Shapes:")
        for item in out:
            if isinstance(item, list):
                for sub_item in item:
                    print(sub_item.shape)
            else:
                print(item.shape)