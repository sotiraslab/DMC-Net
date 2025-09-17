import torch
from torch import nn

from nnunetv2.network.modules3D import StackedConvBlocks, StackedDMRCBlocks, StackedDMSCBlocks, StackedDMCBlocks

class Encoder3D(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage):
        super().__init__()

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if s != 0:
                stage_modules.append(nn.MaxPool3d(kernel_size=2, stride=2))
            stage_modules.append(StackedDMSCBlocks(input_channels, features_per_stage[s]))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        return ret



if __name__ == '__main__':
    #data = torch.rand((1, 4, 128, 128, 128))
    #model = PlainMSConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
    #                            (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)
    #print(model.compute_conv_feature_map_size(data.shape[2:]))

    data = torch.rand((1, 32, 512, 512)).to("cuda")

    model = Encoder3D(32, 6, (32, 64, 128, 256, 512, 512), nn.Conv2d, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), True)

    #import hiddenlayer as hl
    #from torchinfo import summary

    #summary(model, (16, 32, 512, 512))

    #g = hl.build_graph(model, data, transforms=None)
    #g.save("network_architecture.pdf")
    #del g

    #print(model.compute_conv_feature_map_size(data.shape[2:]))