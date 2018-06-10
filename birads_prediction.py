import tensorflow as tf
import numpy
from scipy import misc

import models


def optimistic_restore(session, save_file):
    """
    Function that loads saved parameters
    :param session:
    :param save_file:
    :return:
    """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted(
        [(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
    restore_vars = []

    with tf.variable_scope('', reuse=True):

        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()

            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)

    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def load_images(image_path, view):
    """
    Function that loads and preprocess input images
    :param image_path:
    :param view:
    :return:
    """

    def normalise_single_image(image):
        image -= numpy.mean(image)
        image /= numpy.maximum(numpy.std(image), 10 ** (-5))

    image = misc.imread(image_path + view + '.png')
    image = image.astype(numpy.float32)
    normalise_single_image(image)
    image = numpy.expand_dims(image, axis=0)
    image = numpy.expand_dims(image, axis=3)

    return image


def sample_prediction(parameters):
    """
    Function that creates a model, loads the parameters, and makes a prediction
    :param parameters:
    :return:
    """
    tf.set_random_seed(7)

    with tf.device('/' + parameters['device_type']):
        # initialize input holders
        x_L_CC = tf.placeholder(tf.float32,
                                shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
        x_R_CC = tf.placeholder(tf.float32,
                                shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
        x_L_MLO = tf.placeholder(tf.float32,
                                 shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
        x_R_MLO = tf.placeholder(tf.float32,
                                 shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
        x = (x_L_CC, x_R_CC, x_L_MLO, x_R_MLO)

        # holders for dropout and Guassian noise
        nodropout_probability = tf.placeholder(tf.float32, shape=())
        Gaussian_noise_std = tf.placeholder(tf.float32, shape=())

        # construct models
        model = models.BaselineBreastModel(parameters, x, nodropout_probability, Gaussian_noise_std)
        y_prediction_birads = model.y_prediction_birads

    # allocate computation resources
    if parameters['device_type'] == 'gpu':
        session_config = tf.ConfigProto()
        session_config.gpu_options.visible_device_list = str(parameters['gpu_number'])
    elif parameters['device_type'] == 'cpu':
        session_config = tf.ConfigProto(device_count={'GPU': 0})

    # create a tf session
    session = tf.Session(config=session_config)
    session.run(tf.global_variables_initializer())

    # loads the pre-trained parameters if it's provided
    if parameters['initial_parameters'] is not None:
        saver = tf.train.Saver(max_to_keep=None)
        saver.restore(session, save_path=parameters['initial_parameters'])

    # load input images
    datum_L_CC = load_images(parameters['image_path'], 'L-CC')
    datum_R_CC = load_images(parameters['image_path'], 'R-CC')
    datum_L_MLO = load_images(parameters['image_path'], 'L-MLO')
    datum_R_MLO = load_images(parameters['image_path'], 'R-MLO')

    # populate feed_dict for TF session
    feed_dict_by_model = {nodropout_probability: 1.0, Gaussian_noise_std: 0.0}
    feed_dict_by_model[x_L_CC] = datum_L_CC
    feed_dict_by_model[x_R_CC] = datum_R_CC
    feed_dict_by_model[x_L_MLO] = datum_L_MLO
    feed_dict_by_model[x_R_MLO] = datum_R_MLO

    # run the session for a prediction
    prediction_birads = session.run(y_prediction_birads, feed_dict=feed_dict_by_model)
    birads0_prob = prediction_birads[0][0]
    birads1_prob = prediction_birads[0][1]
    birads2_prob = prediction_birads[0][2]

    # nicely prints out the predictions
    print('BIRADS prediction:\n' +
          '\tBIRADS 0:\t\t\t' + str(birads0_prob) + '\n' +
          '\tBIRADS 1:\t\t\t' + str(birads1_prob) + '\n' +
          '\tBIRADS 2:\t\t\t' + str(birads2_prob))


if __name__ == "__main__":

    # global settings
    parameters = dict(
        device_type='gpu',
        # specify which GPU for training
        gpu_number=1,
        # size of the input images
        input_size=(2600, 2000),
        # path to the input images
        image_path='images/',
        # directory to the saved parameters
        initial_parameters = 'saved_models/model.ckpt'
    )

    # do a sample prediction
    sample_prediction(parameters)