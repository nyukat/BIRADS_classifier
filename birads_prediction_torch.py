import argparse
import torch

import utils
import models_torch as models


def inference(parameters, verbose=True):
    """
    Function that creates a model, loads the parameters, and makes a prediction
    :param parameters: dictionary of parameters
    :param verbose: Whether to print predicted probabilities
    :return: Predicted probabilities for each class
    """
    # resolve device
    device = torch.device(
        "cuda:{}".format(parameters["gpu_number"]) if parameters["device_type"] == "gpu"
        else "cpu"
    )

    # construct models
    model = models.BaselineBreastModel(device, nodropout_probability=1.0, gaussian_noise_std=0.0).to(device)
    model.load_state_dict(torch.load(parameters["model_path"]))

    # load input images and prepare data
    datum_l_cc = utils.load_images(parameters['image_path'], 'L-CC')
    datum_r_cc = utils.load_images(parameters['image_path'], 'R-CC')
    datum_l_mlo = utils.load_images(parameters['image_path'], 'L-MLO')
    datum_r_mlo = utils.load_images(parameters['image_path'], 'R-MLO')
    x = {
        "L-CC": torch.Tensor(datum_l_cc).permute(0, 3, 1, 2).to(device),
        "L-MLO": torch.Tensor(datum_l_mlo).permute(0, 3, 1, 2).to(device),
        "R-CC": torch.Tensor(datum_r_cc).permute(0, 3, 1, 2).to(device),
        "R-MLO": torch.Tensor(datum_r_mlo).permute(0, 3, 1, 2).to(device),
    }

    # run prediction
    with torch.no_grad():
        prediction_birads = model(x).cpu().numpy()

    if verbose:
        # nicely prints out the predictions
        birads0_prob = prediction_birads[0][0]
        birads1_prob = prediction_birads[0][1]
        birads2_prob = prediction_birads[0][2]
        print('BI-RADS prediction:\n' +
              '\tBI-RADS 0:\t' + str(birads0_prob) + '\n' +
              '\tBI-RADS 1:\t' + str(birads1_prob) + '\n' +
              '\tBI-RADS 2:\t' + str(birads2_prob))

    return prediction_birads[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Inference')
    parser.add_argument('--model-path', default='saved_models/model.p')
    parser.add_argument('--device-type', default="cpu")
    parser.add_argument('--gpu-number', default=0, type=int)
    parser.add_argument('--image-path', default="images/")
    args = parser.parse_args()

    parameters_ = {
        "model_path": args.model_path,
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "image_path": args.image_path,
        "input_size": (2600, 2000),
    }

    # do a sample prediction
    inference(parameters_)
