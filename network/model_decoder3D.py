import torch
from torch import nn

from nnunetv2.network.modules3D import StackedConvBlocks, StackedDMRCBlocks, StackedDMSCBlocks, StackedDMCBlocks

class Decoder3D(nn.Module):
    def __init__(self, encoder, num_classes, deep_supervision):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        stages = []
        upsamplings = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]

            upsamplings.append(nn.ConvTranspose3d(
                in_channels=input_features_below, out_channels=input_features_skip, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
            ))
            stages.append(StackedDMSCBlocks(input_features_skip * 2, input_features_skip))
            seg_layers.append(nn.Conv3d(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.upsamplings = nn.ModuleList(upsamplings)
        self.stages = nn.ModuleList(stages)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsamplings[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), dim=1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r


