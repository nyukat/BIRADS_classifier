import argparse
import numpy as np


GOLDEN = (0.21831559, 0.38092783, 0.4007566)


def get_tf_cpu():
    import birads_prediction_tf
    return birads_prediction_tf.inference({
        "model_path": 'saved_models/model.ckpt',
        "device_type": "cpu",
        "gpu_number": 0,
        "image_path": "images/",
        "input_size": (2600, 2000),
    }, verbose=False)


def get_tf_gpu():
    import birads_prediction_tf
    return birads_prediction_tf.inference({
        "model_path": 'saved_models/model.ckpt',
        "device_type": "gpu",
        "gpu_number": 0,
        "image_path": "images/",
        "input_size": (2600, 2000),
    }, verbose=False)


def get_torch_cpu():
    import birads_prediction_torch
    return birads_prediction_torch.inference({
        "model_path": 'saved_models/model.p',
        "device_type": "cpu",
        "gpu_number": 0,
        "image_path": "images/",
        "input_size": (2600, 2000),
    }, verbose=False)


def get_torch_gpu():
    import birads_prediction_torch
    return birads_prediction_torch.inference({
        "model_path": 'saved_models/model.p',
        "device_type": "gpu",
        "gpu_number": 0,
        "image_path": "images/",
        "input_size": (2600, 2000),
    }, verbose=False)


def test_tf_golden_equal():
    assert np.allclose(get_tf_cpu(), GOLDEN)


def test_torch_golden_equal():
    assert np.allclose(get_torch_cpu(), GOLDEN)


def test_tf_cpu_gpu_equal():
    assert np.allclose(get_tf_cpu(), get_tf_gpu())


def test_torch_cpu_gpu_equal():
    assert np.allclose(get_torch_cpu(), get_torch_gpu())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Tests')
    parser.add_argument('--using')
    parser.add_argument('--with-gpu', action="store_true")
    args = parser.parse_args()

    test_list = []
    if args.using == "tf":
        test_list.append(test_tf_golden_equal)
        if args.with_gpu:
            test_list.append(test_tf_cpu_gpu_equal)
    elif args.using == "torch":
        test_list.append(test_torch_golden_equal)
        if args.with_gpu:
            test_list.append(test_torch_cpu_gpu_equal)
    else:
        raise RuntimeError("Provide --using 'tf' or 'torch'")

    for test_func in test_list:
        try:
            test_func()
            print("{}: PASSED".format(test_func.__name__))
        except Exception as e:
            print("{}: FAILED".format(test_func.__name__))
            raise

    print("All {} test(s) passed.".format(len(test_list)))
