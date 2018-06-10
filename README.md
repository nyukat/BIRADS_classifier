# High-resolution breast cancer screening with multi-view deep convolutional neural networks
## Introduction
This is an implementation of the model used for [BI-RADS](https://breast-cancer.ca/bi-rads/) classification as described in our paper ["High-resolution breast cancer screening with multi-view deep convolutional neural networks"](https://arxiv.org/abs/1703.07047). The implementation allows users to get the BI-RADS prediction by applying our pretrained CNN model on standard screening mammogram exam with four views. As a part of this repository, we provide a sample exam (in `images` directory). The model is implemented in both TensorFlow and PyTorch.

## Prerequisites

* Python (3.6)
* TensorFlow (1.5.0) or PyTorch (0.4.0)
* NumPy (1.14.3)
* SciPy (1.0.0)
* Pillow (5.1.0)

## Data

To use the pretrained model, the input is required to consist of four images, one for each view (L-CC, L-MLO, R-CC, R-MLO). Each image has to have the size of 2600x2000 pixels. The images in the provided sample exam were already cropped to the correct size.

## How to run the code
Available options can be found at the bottom of the files `birads_prediction_tf.py` or `birads_prediction_torch.py`.

Run the following command to use the model.

```bash
# Using TensorFlow
python birads_prediction_tf.py

# Using PyTorch
python birads_prediction_torch.py
```

This loads an included sample of four scan views, feeds them into a pretrained copy of our model, and outputs the predicted probabilities of each BI-RADS classification.

You should get the following output:

```
BI-RADS prediction:
        BI-RADS 0:      0.21831559
        BI-RADS 1:      0.38092783
        BI-RADS 2:      0.4007566
```

## Additional options

Additional flags can be provided to the above script:

* `--model-path`: path to a TensorFlow checkpoint or PyTorch pickle of a saved model. By default, this points to the saved model in this repository.
* `--device-type`: whether to use a CPU or GPU. By default, the CPU is used.
* `--gpu-number`: which GPU is used. By default, GPU 0 is used. (Not used if running with CPU)
* `--image-path`: path to saved images. By default, this points to the saved images in this repository. 

For example, to run this script using TensorFlow on GPU 2, run:

```bash
python birads_prediction_tf.py --device-type gpu --gpu-number 2
```

## Converting TensorFlow Models

This repository contains pretrained models in both TensorFlow and PyTorch. The model was originally trained in TensorFlow and translated to PyTorch using the following script:

```bash
python convert_model.py saved_models/model.ckpt saved_models/model.p
```

## Tests

Tests can be configured to your environment.

```bash
# Using TensorFlow, without GPU support
python test_inference.py --using tf

# Using PyTorch, without GPU support
python test_inference.py --using torch

# Using TensorFlow, with GPU support
python test_inference.py --using tf --with-gpu

# Using PyTorch, with GPU support
python test_inference.py --using torch --with-gpu
```

## Reference

If you found this code useful, please cite our paper:

**"High-resolution breast cancer screening with multi-view deep convolutional neural networks"**\
Krzysztof J. Geras, Stacey Wolfson, Yiqiu Shen, Nan Wu, S. Gene Kim, Eric Kim, Laura Heacock, Ujas Parikh, Linda Moy, Kyunghyun Cho\
2017

    @article{geras2017high, 
        title = {High-resolution breast cancer screening with multi-view deep convolutional neural networks},
        author = {Krzysztof J. Geras and Stacey Wolfson and Yiqiu Shen and Nan Wu and S. Gene Kim and Eric Kim and Laura Heacock and Ujas Parikh and Linda Moy and Kyunghyun Cho}, 
        journal = {arXiv:1703.07047},
        year = {2017}
    }
