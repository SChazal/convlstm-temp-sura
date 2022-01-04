from collections import OrderedDict
from ConvRNN import CLSTM_cell


# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
convlstm_encoder_params_large = [
    [
        OrderedDict({'conv1_leaky_1': [5, 64, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [128, 128, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [256, 256, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(73, 144), input_channels=64,
                   filter_size=5, num_features=128),
        CLSTM_cell(shape=(37, 72), input_channels=128,
                   filter_size=5, num_features=256),
        CLSTM_cell(shape=(19, 36), input_channels=256,
                   filter_size=5, num_features=256)
    ]
]

convlstm_decoder_params_large = [
    [
        OrderedDict({'deconv1_leaky_1': [256, 256, (3, 4), 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [256, 256, (3, 4), 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [128, 64, 3, 1, 1],
            'conv4_leaky_1': [64, 5, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(19, 36), input_channels=256,
                   filter_size=5, num_features=256),
        CLSTM_cell(shape=(37, 72), input_channels=256,
                   filter_size=5, num_features=256),
        CLSTM_cell(shape=(73, 144), input_channels=256,
                   filter_size=5, num_features=128),
    ]
]
