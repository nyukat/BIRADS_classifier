import numpy as np
from scipy import misc


def load_images(image_path, view):
    """
    Function that loads and preprocess input images
    :param image_path: base path to image
    :param view: L-CC / R-CC / L-MLO / R-MLO
    :return: Batch x Height x Width x Channels array
    """
    image = misc.imread(image_path + view + '.png')
    image = image.astype(np.float32)
    normalize_single_image(image)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    return image


def normalize_single_image(image):
    """
    Normalize image in-place
    :param image: numpy array
    """
    image -= np.mean(image)
    image /= np.std(image)
