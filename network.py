import config
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.get_convnet_simple()

    def get_convnet_simple(self):
        nn_list = [
            nn.Conv2d(config.args.image_channel, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        ]

        dummy_img = torch.tensor(np.zeros((1, config.args.image_channel, config.args.image_size, config.args.image_size))).float()
        aux_model = nn.Sequential(*nn_list)

        dummy_flatten_vector = aux_model(dummy_img)
        in_features = list(dummy_flatten_vector.shape)[1]
        second_layer_features = 256
        nn_list = nn_list + [
            nn.Linear(in_features, second_layer_features),
            nn.LeakyReLU(),
            nn.Linear(second_layer_features, config.args.output_shape),
            nn.Softmax(dim=1)
        ]

        self.model = nn.Sequential(*nn_list)

    def forward(self, x):
        return self.model(x)
