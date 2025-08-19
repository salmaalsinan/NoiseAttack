import torch
import torch.nn as nn

class BaseUnet(nn.Module):
    def __init__(self, *pargs, activation=None, in_channels=3, out_channels=3, **kwargs):
        super(BaseUnet, self).__init__()
        self.unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', *pargs,
                                   in_channels=in_channels, out_channels=out_channels, **kwargs)
        if activation is None:
            self.activation = nn.Identity()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unknown activation")

    def forward(self, x):
        x = self.unet(x)
        x = self.activation(x)
        return x

