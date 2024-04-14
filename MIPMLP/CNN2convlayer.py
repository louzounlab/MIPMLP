import torch
from torch import nn
from math import floor
from .PLSuperModel import SuperModel


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class CNN(SuperModel):
    def __init__(self, params, mode=None, task="reg", weighted=False):
        """

        @param params: A dictionary in this format:
            params = {
        "l1_loss": 0.1,
        "weight_decay": 0.01,
        "lr": 0.001,
        "batch_size": 128,
        "activation": "elu", | "relu" | "tanh"
        "dropout": 0.1,
        "kernel_size_a": 4,
        "kernel_size_b": 4,
        "stride": 2,
        "padding": 3,
        "padding_2": 0,
        "kernel_size_a_2": 2,
        "kernel_size_b_2": 7,
        "stride_2": 3,
        "channels": 3,
        "channels_2": 14,
        "linear_dim_divider_1": 10,
        "linear_dim_divider_2": 6
    }
    Explanation of the params:
    l1 loss = the coefficient of the L1 loss
    weight decay = L2 regularization
    lr = learning rate
    batch size = as it sounds
    activation = activation function one of:  "elu", | "relu" | "tanh"
    dropout = as it sounds (is common to all the layers)
    kernel_size_a = the size of the kernel of the first CNN layer (rows)
    kernel_size_b = the size of the kernel of the first CNN layer (columns)
    stride = the stride's size of the first CNN
    padding = the padding's size of the first CNN layer
    padding_2 = the padding's size of the second CNN layer
    kernel_size_a_2 = the size of the kernel of the second CNN layer (rows)
    kernel_size_b_2 = the size of the kernel of the second CNN layer (columns)
    stride_2 = the stride's size of the second CNN
    channels = number of channels of the first CNN layer
    channels_2 = number of channels of the second CNN layer
    linear_dim_divider_1 = the number to divide the original input size to get the number of neurons in the first FCN layer
    linear_dim_divider_2 = the number to divide the original input size to get the number of neurons in the second FCN layer

        @param mode: it should be "dendogram" to get iMic
        @param task: one of "reg" or "class"
        @param weighted: default is False, change to True foe weighted BCE
        """
        super().__init__(params, mode, task, weighted)

        in_dim = self.in_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(1, params["channels"], kernel_size=(params["kernel_size_a"], params["kernel_size_b"]),
                      stride=params["stride"], padding=params["padding"]),
            self.activation(),

            nn.Conv2d(params["channels"], params["channels_2"],
                      kernel_size=(params["kernel_size_a_2"], params["kernel_size_b_2"]),
                      stride=params["stride_2"], padding=params["padding_2"]),
            self.activation(),
        )

        add = 0
        if mode is not None:
            in_dim = (in_dim[0], in_dim[1] - mode.shape[1])
            add = mode.shape[1]

        cos1 = conv_output_shape(in_dim, (params["kernel_size_a"], params["kernel_size_b"]), stride=params["stride"],
                                 pad=params["padding"])
        cos = conv_output_shape(cos1, (params["kernel_size_a_2"], params["kernel_size_b_2"]), stride=params["stride_2"],
                                pad=params["padding_2"])

        conv_out_dim = int(cos[0] * cos[1] * params["channels_2"]) + add

        if conv_out_dim > self.threshold:
            self.use_max_pool = True
            max_pool_factor = int(((conv_out_dim - add) // self.threshold) ** 0.5)
            if max_pool_factor <= 1:
                max_pool_factor = 2
            conv_out_dim = (cos[0] // max_pool_factor) * (cos[1] // max_pool_factor) * params["channels_2"] + add
            self.MP = nn.MaxPool2d(max_pool_factor)
        else:
            self.use_max_pool = False

        self.lin = nn.Sequential(
            nn.Linear(conv_out_dim, conv_out_dim // params["linear_dim_divider_1"]),
            self.activation(),
            nn.Dropout(params["dropout"]),
            nn.Linear(conv_out_dim // params["linear_dim_divider_1"], conv_out_dim // params["linear_dim_divider_2"]),
            self.activation(),
            nn.Dropout(params["dropout"]),
            nn.Linear(conv_out_dim // params["linear_dim_divider_2"], 1)
        )

    def forward(self, x, b=None):
        x = x.type(torch.float32)
        x = torch.unsqueeze(x, 1)
        x = self.cnn(x)
        if self.use_max_pool:
            x = self.MP(x)
        x = torch.flatten(x, 1)
        if b is not None:
            x = torch.cat([x, b], dim=1).type(torch.float32)
        x = self.lin[:-1](x)
        self.last_layer = x
        x = self.lin[-1](x)
        x = x.type(torch.float32)
        return x
