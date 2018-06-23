import torch
import torch.nn as nn
import torch.nn.functional as F
import collections as col

import layers_torch as layers


class BaselineBreastModel(nn.Module):
    def __init__(self, nodropout_probability=None, gaussian_noise_std=None):
        super(BaselineBreastModel, self).__init__()
        self.conv_layer_dict = col.OrderedDict()
        self.conv_layer_dict["conv1"] = layers.AllViewsConvLayer(
            1, number_of_filters=32, filter_size=(3, 3), stride=(2, 2),
        )

        self.conv_layer_dict["conv2a"] = layers.AllViewsConvLayer(
            32, number_of_filters=64, filter_size=(3, 3), stride=(2, 2),
        )
        self.conv_layer_dict["conv2b"] = layers.AllViewsConvLayer(
            64, number_of_filters=64, filter_size=(3, 3), stride=(1, 1),
        )
        self.conv_layer_dict["conv2c"] = layers.AllViewsConvLayer(
            64, number_of_filters=64, filter_size=(3, 3), stride=(1, 1),
        )

        self.conv_layer_dict["conv3a"] = layers.AllViewsConvLayer(
            64, number_of_filters=128, filter_size=(3, 3), stride=(1, 1),
        )
        self.conv_layer_dict["conv3b"] = layers.AllViewsConvLayer(
            128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1),
        )
        self.conv_layer_dict["conv3c"] = layers.AllViewsConvLayer(
            128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1),
        )

        self.conv_layer_dict["conv4a"] = layers.AllViewsConvLayer(
            128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1),
        )
        self.conv_layer_dict["conv4b"] = layers.AllViewsConvLayer(
            128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1),
        )
        self.conv_layer_dict["conv4c"] = layers.AllViewsConvLayer(
            128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1),
        )

        self.conv_layer_dict["conv5a"] = layers.AllViewsConvLayer(
            128, number_of_filters=256, filter_size=(3, 3), stride=(1, 1),
        )
        self.conv_layer_dict["conv5b"] = layers.AllViewsConvLayer(
            256, number_of_filters=256, filter_size=(3, 3), stride=(1, 1),
        )
        self.conv_layer_dict["conv5c"] = layers.AllViewsConvLayer(
            256, number_of_filters=256, filter_size=(3, 3), stride=(1, 1),
        )
        self._conv_layer_ls = nn.ModuleList(self.conv_layer_dict.values())

        self.all_views_pad = layers.AllViewsPad()
        self.all_views_max_pool = layers.AllViewsMaxPool()
        self.all_views_avg_pool = layers.AllViewsAvgPool()

        self.fc1 = nn.Linear(256 * 4, 256 * 4)
        self.fc2 = nn.Linear(256 * 4, 3)

        self.gaussian_noise_layer = layers.AllViewsGaussianNoise(gaussian_noise_std)
        self.dropout = nn.Dropout(p=1 - nodropout_probability)

    def forward(self, x):
        x = self.gaussian_noise_layer(x)
        x = self.conv_layer_dict["conv1"](x)
        x = self.all_views_max_pool(x, stride=(3, 3))
        x = self.conv_layer_dict["conv2a"](x)
        x = self.conv_layer_dict["conv2b"](x)
        x = self.conv_layer_dict["conv2c"](x)
        x = self.all_views_max_pool(x, stride=(2, 2))
        x = self.conv_layer_dict["conv3a"](x)
        x = self.conv_layer_dict["conv3b"](x)
        x = self.conv_layer_dict["conv3c"](x)
        x = self.all_views_pad(x, pad=(0, 1, 0, 0))
        x = self.all_views_max_pool(x, stride=(2, 2))
        x = self.conv_layer_dict["conv4a"](x)
        x = self.conv_layer_dict["conv4b"](x)
        x = self.conv_layer_dict["conv4c"](x)
        x = self.all_views_max_pool(x, stride=(2, 2))
        x = self.conv_layer_dict["conv5a"](x)
        x = self.conv_layer_dict["conv5b"](x)
        x = self.conv_layer_dict["conv5c"](x)
        x = self.all_views_avg_pool(x)

        x = torch.cat([
            x["L-CC"],
            x["R-CC"],
            x["L-MLO"],
            x["R-MLO"],
        ], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x


class BaselineHistogramModel(nn.Module):

    def __init__(self, num_bins):
        super(BaselineHistogramModel, self).__init__()
        self.fc1 = nn.Linear(num_bins * 4, 100)
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    def param_dict(self):
        return dict(zip(
            ["w0", "b0", "w1", "b1"],
            self.parameters(),
        ))