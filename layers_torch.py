import torch
import torch.nn as nn
import torch.nn.functional as F


class AllViewsGaussianNoise(nn.Module):
    def __init__(self, gaussian_noise_std):
        super(AllViewsGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, x):
        if self.gaussian_noise_std:
            return x

        device = x["L-CC"].get_device() if x["L-CC"].is_cuda else torch.device("cpu")

        return {
            k: v + torch.Tensor(*v.shape).normal_(std=self.gaussian_noise_std).to(device)
            for k, v in x.items()
        }


class AllViewsConvLayer(nn.Module):
    def __init__(self, in_channels, number_of_filters=32, filter_size=(3, 3), stride=(1, 1)):
        super(AllViewsConvLayer, self).__init__()
        self.cc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=number_of_filters,
            kernel_size=filter_size,
            stride=stride,
        )
        self.mlo = nn.Conv2d(
            in_channels=in_channels,
            out_channels=number_of_filters,
            kernel_size=filter_size,
            stride=stride,
        )

    def forward(self, x):
        return {
            "L-CC": F.relu(self.cc(x["L-CC"])),
            "L-MLO": F.relu(self.mlo(x["L-MLO"])),
            "R-CC": F.relu(self.cc(x["R-CC"])),
            "R-MLO": F.relu(self.mlo(x["R-MLO"])),
        }

    @property
    def ops(self):
        return {
            "CC": self.cc,
            "MLO": self.mlo,
        }


class AllViewsMaxPool(nn.Module):
    def __init__(self):
        super(AllViewsMaxPool, self).__init__()

    def forward(self, x, stride=(2, 2), padding=(0, 0)):
        return {
            "L-CC": F.max_pool2d(
                x["L-CC"], kernel_size=stride, stride=stride, padding=padding),
            "L-MLO": F.max_pool2d(
                x["L-MLO"], kernel_size=stride, stride=stride, padding=padding),
            "R-CC": F.max_pool2d(
                x["R-CC"], kernel_size=stride, stride=stride, padding=padding),
            "R-MLO": F.max_pool2d(
                x["R-MLO"], kernel_size=stride, stride=stride, padding=padding),
        }


class AllViewsAvgPool(nn.Module):
    def __init__(self):
        super(AllViewsAvgPool, self).__init__()

    def forward(self, x):
        return {
            k: v.view(v.size()[:2] + (-1,)).mean(-1)
            for k, v in x.items()
        }


class AllViewsPad(nn.Module):
    def __init__(self):
        super(AllViewsPad, self).__init__()

    def forward(self, x, pad):
        return {
            k: F.pad(v, pad)
            for k, v in x.items()
        }
