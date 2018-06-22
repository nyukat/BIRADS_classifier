import numpy as np
from scipy import misc


def load_images(image_path, view):
    """
    Function that loads and preprocess input images
    :param image_path:
    :param view:
    :return:
    """

    def normalise_single_image(image_):
        image_ -= np.mean(image_)
        image_ /= np.std(image_)

    image = misc.imread(image_path + view + '.png')
    image = image.astype(np.float32)
    normalise_single_image(image)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    return image
