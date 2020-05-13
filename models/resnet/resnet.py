import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet.residual_block import ResidualBlock
from util import WNConv2d


class ResNet(nn.Module):
    """ResNet for scale and translate factors in Real NVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        out_channels (int): Number of channels in the output.
        num_blocks (int): Number of residual blocks in the network.
        kernel_size (int): Side length of each filter in convolutional layers.
        padding (int): Padding for convolutional layers.
        double_after_norm (bool): Double input after input BatchNorm.
    """
    def __init__(self, in_channels, mid_channels, out_channels,
                 num_blocks, kernel_size, padding, double_after_norm):
        super(ResNet, self).__init__()
        self.in_norm = nn.BatchNorm2d(in_channels)
        self.double_after_norm = double_after_norm
        self.in_conv = WNConv2d(2 * in_channels, mid_channels, kernel_size, padding, bias=True)
        self.in_skip = WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)

        self.blocks = nn.ModuleList([ResidualBlock(mid_channels, mid_channels)
                                     for _ in range(num_blocks)])
        self.skips = nn.ModuleList([WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)
                                    for _ in range(num_blocks)])

        self.out_norm = nn.BatchNorm2d(mid_channels)
        self.out_conv = WNConv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # try:
        x = self.in_norm(x) # FIND THIS MF
        # except RuntimeError:
        # 	import ipdb; ipdb.set_trace()
        if self.double_after_norm:
            x *= 2.
        # WORTH ATTENTION: import ipdb; ipdb.set_trace()
        x = torch.cat((x, -x), dim=1)
        x = F.relu(x)
        # try:
        x = self.in_conv(x)
        # except:
        # 	import ipdb; ipdb.set_trace()
        # WHY DOES THIS EXIST?
        x_skip = self.in_skip(x)

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x) # TODO: try without self.skip() and inspect performance.
            '''we'd like to change it to:'''
            # x += skip(x) # TODO: try without self.skip() and inspect performance.

        x = self.out_norm(x_skip)
        '''we'd like to change it to: (read * explanation down below)'''
        # x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        return x

''' * explanation:
Skip connections are already made in residual_block.py, on
a per-block basis. Yet, in the for loop (line 56) we are using
increment operator '+=' to add all block results to `x_skip`.
This is then fed to self.out_norm in line 60.
While in doubt, the net works. We hereby choose to save this for later checking. 
'''
