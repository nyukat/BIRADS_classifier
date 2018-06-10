import layers_tf as layers


def baseline(x, parameters, nodropout_probability=None, gaussian_noise_std=None):

    if gaussian_noise_std is not None:
        x = layers.all_views_gaussian_noise_layer(x, gaussian_noise_std)

    # first conv sequence
    h = layers.all_views_conv_layer(x, 'conv1', number_of_filters=32, filter_size=[3, 3], stride=[2, 2])

    # second conv sequence
    h = layers.all_views_max_pool(h, stride=[3, 3])
    h = layers.all_views_conv_layer(h, 'conv2a', number_of_filters=64, filter_size=[3, 3], stride=[2, 2])
    h = layers.all_views_conv_layer(h, 'conv2b', number_of_filters=64, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, 'conv2c', number_of_filters=64, filter_size=[3, 3], stride=[1, 1])

    # third conv sequence
    h = layers.all_views_max_pool(h, stride=[2, 2])
    h = layers.all_views_conv_layer(h, 'conv3a', number_of_filters=128, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, 'conv3b', number_of_filters=128, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, 'conv3c', number_of_filters=128, filter_size=[3, 3], stride=[1, 1])

    # fourth conv sequence
    h = layers.all_views_max_pool(h, stride=[2, 2])
    h = layers.all_views_conv_layer(h, 'conv4a', number_of_filters=128, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, 'conv4b', number_of_filters=128, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, 'conv4c', number_of_filters=128, filter_size=[3, 3], stride=[1, 1])

    # fifth conv sequence
    h = layers.all_views_max_pool(h, stride=[2, 2])
    h = layers.all_views_conv_layer(h, 'conv5a', number_of_filters=256, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, 'conv5b', number_of_filters=256, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, 'conv5c', number_of_filters=256, filter_size=[3, 3], stride=[1, 1])

    # Pool, flatten, and fully connected layers
    h = layers.all_views_global_avg_pool(h)
    h = layers.all_views_flattening_layer(h)
    h = layers.fc_layer(h, number_of_units=1024)
    h = layers.dropout_layer(h, nodropout_probability)

    y_prediction_birads = layers.softmax_layer(h, number_of_outputs=3)

    return y_prediction_birads


class BaselineBreastModel:

    def __init__(self, parameters, x, nodropout_probability=None, gaussian_noise_std=None):
        self.y_prediction_birads = baseline(x, parameters, nodropout_probability, gaussian_noise_std)
