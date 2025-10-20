from typing import Tuple, Union, List, Type
import torch
import torch.nn as nn
import os
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BasicBlockD
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

class ResEncoderUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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

    def forward(self, x):

        conv_enc_outputs = [self.conv_encoder_blocks[0](x)]
        for i in range(1, len(self.conv_encoder_blocks)):
            conv_enc_outputs.append(self.conv_encoder_blocks[i](conv_enc_outputs[-1]))

        lres_input = conv_enc_outputs[-1]
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

        return r

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = torch.randn(size=(2, 1, 256, 256, 128)).to(device)
    
    net = ResEncoderUNet(
        in_channels=1,
        out_channels=6,
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
