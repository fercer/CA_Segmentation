import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d] and m.weight.requires_grad:
        torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
            torch.nn.init.zeros_(m.bias)


class UNet(nn.Module):
    def __init__(self, input_channels, output_classes, hidden_dim=64, multiplier=2, depth=3):
        super(UNet, self).__init__()
        self._input_layer = nn.Sequential(nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1, bias=True),
                                          nn.ReLU(),
                                          nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True),
                                          nn.ReLU()
                                          )

        encoder_path = []
        d = 0
        for _ in range(depth):
            encoder_path.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(hidden_dim * multiplier ** d, hidden_dim * multiplier ** (d+1), kernel_size=3, padding=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(hidden_dim * multiplier ** (d+1), hidden_dim * multiplier ** (d+1), kernel_size=3, padding=1, bias=True),
                nn.ReLU()
            ))
            d += 1

        self._encoder_path = nn.ModuleList(encoder_path)
        self._bottleneck = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(hidden_dim * multiplier ** d, hidden_dim * multiplier ** (d+1), kernel_size=3, padding=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(hidden_dim * multiplier ** (d+1), hidden_dim * multiplier ** (d+1), kernel_size=3, padding=1, bias=True),
                nn.ReLU(),
                nn.ConvTranspose2d(hidden_dim * multiplier ** (d + 1), hidden_dim * multiplier ** d, kernel_size=2, stride=2, bias=True)
        )
        d+=1

        decoder_path = []
        for _ in range(depth):
            decoder_path.append(nn.Sequential(
                nn.Conv2d(hidden_dim * multiplier ** d, hidden_dim * multiplier ** (d-1), kernel_size=3, padding=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(hidden_dim * multiplier ** (d-1), hidden_dim * multiplier ** (d-1), kernel_size=3, padding=1, bias=True),
                nn.ReLU(),
                nn.ConvTranspose2d(hidden_dim * multiplier ** (d-1), hidden_dim * multiplier ** (d-2), kernel_size=2, stride=2, bias=True)
            ))
            d -= 1
        d -= 1

        self._decoder_path = nn.ModuleList(decoder_path)

        self._classifier = nn.Sequential(
            nn.Conv2d(hidden_dim * multiplier ** (d+1), hidden_dim * multiplier ** d, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * multiplier ** d, hidden_dim * multiplier ** d, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * multiplier ** d, output_classes, kernel_size=1, bias=True)
        )

        self.apply(init_weights)

    def forward(self, x):
        fx = self._input_layer(x)

        x_stack = [fx]
        for layer in self._encoder_path:
            fx = layer(fx)
            x_stack.append(fx)

        fx = self._bottleneck(fx)

        for layer in self._decoder_path:
            fx = layer(torch.cat((fx, x_stack.pop()), dim=1))

        fx = self._classifier(torch.cat((fx, x_stack.pop()), dim=1))
        return fx


if __name__ == '__main__':

    unet = UNet(1, 2, 64, 2, 4)

    inputs = torch.rand(2, 1, 64, 64)

    resp = unet(inputs)

    print('Resp size:', resp.size())
