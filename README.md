# High-resolution breast cancer screening with multi-view deep convolutional neural networks
## Introduction
This is an implementation of the model used for [BI-RADS](https://breast-cancer.ca/bi-rads/) classification as described in our paper ["High-resolution breast cancer screening with multi-view deep convolutional neural networks"]https://arxiv.org/abs/1703.07047. The implementation allows users to get the BI-RADS prediction by applying our pretrained CNN model on standrad screening mammogram exam with four views. As a part of this repository, we provide a sample exam (in `images` directory).

## Prerequisites

* Python (3.6), TensorFlow (1.5.0), NumPy (1.14.3), SciPy (1.0.0)
* NVIDIA GPU (we used Tesla M40).

## Data

To use the pretrained model, the input is required to consist of four images, one for each view (L-CC, L-MLO, R-CC, R-MLO). Each image has to have the size of 2600x2000 pixels. The images in the provided sample exam were already cropped to the correct size.

## How to run the code
Available options can be found at the bottom of the file `birads_prediction.py`. Please keep `input_size = (2600, 2000)` as the provided pretrained models were trained with images in this resolution. You may need to change `gpu_number`.

Run the following command to use the model.

```bash
python birads_prediction.py
```

#TODO: the results people should get when running the model. Please look at the other repository for what it should look like.

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
