import torch.nn
from torch import nn

import torch.nn.functional as F

class StackedDMCBlocks(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int
                 ):

        super().__init__()

        self.convs = nn.Sequential(
            DMSC(input_channels, output_channels),
            DMRC(output_channels, output_channels)
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class DMRC(nn.Module):
    def __init__(self,
                 input_channels: int, 
                 output_channels: int
                 ):
        super(DMRC, self).__init__()

        self.ge = nn.Sequential(
                    nn.AvgPool2d(kernel_size=4, stride=4), 
                    #nn.AvgPool2d(kernel_size=2, stride=2), 
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
                    )

        self.att_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)

        self.nonlin = nn.Sigmoid()

        self.conv = nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

        self.next_conv = nn.Sequential(
                    nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(output_channels, output_channels), 
            nn.Sigmoid()
        )

    def forward(self, x):
        att1 = F.interpolate(self.ge(x), x.size()[2:])
        att2 = self.att_conv(x)
        att = torch.add(att1, att2)
        att = self.nonlin(att)
        output = self.conv(x)
        output = output * att
        output = self.next_conv(output)
        ch_att = self.fc(self.avg_pool(output).squeeze()).unsqueeze(dim=-1).unsqueeze(dim=-1)
        output = output * ch_att
        return output


class StackedDMRCBlocks(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int
                 ):

        super().__init__()

        self.convs = DMRC(input_channels, output_channels)

        self.conv1 = nn.Sequential(
                    nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

    def forward(self, x):
        x = self.convs(x)
        x = self.conv1(x)
        return x


class DMSC(nn.Module):
    def __init__(self,
                 input_channels: int, 
                 output_channels: int
                 ):
        super(DMSC, self).__init__()

        self.conv1 = nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels),
                    nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

        self.conv2 = nn.Sequential(
                    #nn.Conv2d(input_channels, output_channels, kernel_size=1),
                    #nn.Conv2d(output_channels, output_channels, kernel_size=5, stride=1, padding=2, groups=output_channels),
                    nn.Conv2d(input_channels, output_channels, kernel_size=5, stride=1, padding=2),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels),
                    #nn.Conv2d(output_channels, output_channels, kernel_size=5, stride=1, padding=2, groups=output_channels),
                    nn.Conv2d(output_channels, output_channels, kernel_size=5, stride=1, padding=2),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(output_channels * 2, output_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        output = torch.cat([x1, x2], dim=1)
        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        #output = output + x
        return output

class StackedDMSCBlocks(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int
                 ):

        super().__init__()

        self.convs = DMSC(input_channels, output_channels)

        self.conv1 = nn.Sequential(
                    nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

    def forward(self, x):
        x = self.convs(x)
        x = self.conv1(x)
        return x

class Conv(nn.Module):
    def __init__(self,
                input_channels,
                output_channels
                ):
        super(Conv, self).__init__()

        self.conv = (nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.BatchNorm2d(output_channels)
            ))

    def forward(self, x):
        x = self.conv(x)
        return x

class StackedConvBlocks(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int
                 ):

        super().__init__()

        self.convs = nn.Sequential(
            Conv(input_channels, output_channels),
            Conv(output_channels, output_channels)
        )

    def forward(self, x):
        x = self.convs(x)
        return x




class StackedDMCCFIBlocks(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 cardinality: int
                 ):

        super().__init__()

        self.convs = nn.Sequential(
            DMSC_CFI(input_channels, output_channels, cardinality),
            DMRC_CFI(output_channels, output_channels, cardinality)
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class DMRC_CFI(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 cardinality
                 ):
        super(DMRC_CFI, self).__init__()

        self.ge = nn.Sequential(
                    nn.AvgPool2d(kernel_size=4, stride=4), 
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
                    )

        self.att_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)

        self.nonlin = nn.Sigmoid()

        self.conv = nn.Sequential(
                    CFI(input_channels=input_channels, output_channels=output_channels, kernel_size=3, stride=1, padding=1, cardinality=cardinality),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

        self.next_conv = nn.Sequential(
                    CFI(input_channels=output_channels, output_channels=output_channels, kernel_size=3, stride=1, padding=1, cardinality=cardinality),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(output_channels, output_channels), 
            nn.Sigmoid()
        )

    def forward(self, x):
        att1 = F.interpolate(self.ge(x), x.size()[2:])
        att2 = self.att_conv(x)
        att = torch.add(att1, att2)
        att = self.nonlin(att)
        output = self.conv(x)
        output = output * att
        output = self.next_conv(output)
        ch_att = self.fc(self.avg_pool(output).squeeze()).unsqueeze(dim=-1).unsqueeze(dim=-1)
        output = output * ch_att
        return output


class StackedDMRCCFIBlocks(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 cardinality
                 ):

        super().__init__()

        self.convs = DMRC_CFI(input_channels, output_channels, cardinality)

        self.conv1 = nn.Sequential(
                    nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

    def forward(self, x):
        x = self.convs(x)
        x = self.conv1(x)
        return x



class DMSC_CFI(nn.Module):
    def __init__(self,
                 input_channels, 
                 output_channels,
                 cardinality
                 ):
        super(DMSC_CFI, self).__init__()

        self.conv1 = nn.Sequential(
                    CFI(input_channels=input_channels, output_channels=output_channels, kernel_size=3, stride=1, padding=1, cardinality=cardinality),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels),
                    CFI(input_channels=output_channels, output_channels=output_channels, kernel_size=3, stride=1, padding=1, cardinality=cardinality),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

        self.conv2 = nn.Sequential(
                    CFI(input_channels=input_channels, output_channels=output_channels, kernel_size=5, stride=1, padding=2, cardinality=cardinality),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels),
                    CFI(input_channels=output_channels, output_channels=output_channels, kernel_size=5, stride=1, padding=2, cardinality=cardinality),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(output_channels * 2, output_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        output = torch.cat([x1, x2], dim=1)
        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        #output = output + x
        return output

class StackedDMSCCFIBlocks(nn.Module):
    def __init__(self,
                input_channels,
                output_channels,
                cardinality
                 ):

        super().__init__()

        self.convs = DMSC_CFI(input_channels, output_channels, cardinality)

        self.conv1 = nn.Sequential(
                    nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

    def forward(self, x):
        x = self.convs(x)
        x = self.conv1(x)
        return x


class Conv_CFI(nn.Module):
    def __init__(self,
                input_channels,
                output_channels,
                cardinality
                ):
        super(Conv_CFI, self).__init__()

        self.conv = (nn.Sequential(
                CFI(input_channels=input_channels, output_channels=output_channels, kernel_size=3, stride=1, padding=1, cardinality=cardinality),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.BatchNorm2d(output_channels)
            ))

    def forward(self, x):
        x = self.conv(x)
        return x

class StackedConvCFIBlocks(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 cardinality
                 ):

        super().__init__()

        self.convs = nn.Sequential(
            Conv_CFI(input_channels, output_channels, cardinality),
            Conv_CFI(output_channels, output_channels, cardinality)
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class CFI(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 stride,
                 padding,
                 cardinality
                 ):
        super(CFI, self).__init__()

        convs = []
        #if cardinality == 1:
        #    self.conv = nn.Conv2d(
        #        in_channels=input_channels,
        #        out_channels=output_channels,
        #        kernel_size=kernel_size,
        #        stride=stride,
        #        padding=padding,
        #        bias=True)
        #else:
        for c in range(cardinality):
            self.conv = nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels // cardinality,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=True
                )
            convs.append(self.conv)
            
        self.convs = nn.Sequential(*convs)

    def forward(self, x):

        output = self.convs[0](x)
        for c in range(1, len(self.convs)):
            output = torch.cat((output, self.convs[c](x)), dim=1)
        
        return output
