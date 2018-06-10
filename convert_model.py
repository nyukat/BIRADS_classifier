import argparse
import torch
import tensorflow as tf

import models_torch


def tf_to_torch(input_path, output_path):
    """
    Convert TensorFlow checkpoint to PyTorch model pickle
    :param input_path: path to TensorFlow checkpoint
    :param output_path: path to save PyTorch model pickle
    """
    g = tf.Graph()
    device = torch.device("cpu")
    bbmodel = models_torch.BaselineBreastModel(device, nodropout_probability=1.0)
    with tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.import_meta_graph(input_path + ".meta")
        saver.restore(sess, input_path)
        var_dict = {
            var.name: var
            for var in g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        }
        for conv_name, conv_layer in bbmodel.conv_layer_dict.items():
            for view in ["CC", "MLO"]:
                conv_layer.ops[view].weight.data = torch.Tensor(sess.run(
                    var_dict["{}_{}/weights:0".format(conv_name, view)]
                )).permute(3, 2, 0, 1)
                conv_layer.ops[view].bias.data = torch.Tensor(sess.run(
                    var_dict["{}_{}/biases:0".format(conv_name, view)]
                ))
        bbmodel.fc1.weight.data = torch.Tensor(sess.run(
            var_dict["fully_connected/weights:0"]
        ).T)
        bbmodel.fc1.bias.data = torch.Tensor(sess.run(
            var_dict["fully_connected/biases:0"]
        ))
        bbmodel.fc2.weight.data = torch.Tensor(sess.run(
            var_dict["fully_connected_1/weights:0"]
        ).T)
        bbmodel.fc2.bias.data = torch.Tensor(sess.run(
            var_dict["fully_connected_1/biases:0"]
        ))
        torch.save(bbmodel.state_dict(), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert from TensorFlow checkpoints to PyTorch pickles')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    tf_to_torch(args.input_path, args.output_path)
