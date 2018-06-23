import birads_prediction_tf
import birads_prediction_torch

import numpy as np


GOLDEN = (0.21831559, 0.38092783, 0.4007566)


def get_tf_cpu():
    return birads_prediction_tf.inference({
        "model_path": 'saved_models/model.ckpt',
        "device_type": "cpu",
        "gpu_number": 0,
        "image_path": "images/",
        "input_size": (2600, 2000),
    })


def get_tf_gpu():
    return birads_prediction_tf.inference({
        "model_path": 'saved_models/model.ckpt',
        "device_type": "gpu",
        "gpu_number": 0,
        "image_path": "images/",
        "input_size": (2600, 2000),
    })


def get_torch_cpu():
    return birads_prediction_torch.inference({
        "model_path": 'saved_models/model.p',
        "device_type": "cpu",
        "gpu_number": 0,
        "image_path": "images/",
        "input_size": (2600, 2000),
    })


def get_torch_gpu():
    return birads_prediction_torch.inference({
        "model_path": 'saved_models/model.p',
        "device_type": "gpu",
        "gpu_number": 0,
        "image_path": "images/",
        "input_size": (2600, 2000),
    })


def test_tf_golden_equal():
    assert np.allclose(get_tf_cpu(), GOLDEN)


def test_torch_golden_equal():
    assert np.allclose(get_torch_cpu(), GOLDEN)


def test_tf_cpu_gpu_equal():
    assert np.allclose(get_tf_cpu(), get_tf_gpu())


def test_torch_cpu_gpu_equal():
    assert np.allclose(get_torch_cpu(), get_torch_gpu())
