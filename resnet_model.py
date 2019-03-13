from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Add, \
    Activation, GlobalAveragePooling2D
from keras import regularizers

cardinality = 8
bottle_width = 64
chan_per_group = 8  # 64 / 8
num_blocks = [4, 6, 3]
momentum = 0.9
epsilon = 1e-5
weight_decay = 5e-4


def conv_layer(x, filters, kernel_size, strides, name):
    """conv2D layer"""
    return Conv2D(
        filters, kernel_size,
        strides=strides, padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=name
    )(x)


def bn_relu_conv(x, filters, kernel_size, strides, name):
    """pre-activation conv2D block"""

    x = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_bn')(x)
    x = Activation(activation='relu', name=name + '_relu')(x)
    x = conv_layer(x, filters, kernel_size, strides, name + '_conv2D')
    return x


def res_block(x, dim, name):
    """residue block: pre-activation"""

    # main path
    identity = x
    if int(x.shape[-1]) != dim:
        identity = conv_layer(identity, dim, (1, 1), (2, 2), name + '_main_DS')

    # residual path
    # 1X1 bottleneck
    res = x
    if int(x.shape[-1]) != dim:
        res = bn_relu_conv(res, dim // 4, (1, 1), (2, 2), name + '_bottle')
    else:
        res = bn_relu_conv(res, dim // 4, (1, 1), (1, 1), name + '_bottle')

    # 3X3 conv2D
    res = bn_relu_conv(res, dim // 4, (3, 3), (1, 1), name + '_3x3_conv')

    # 1X1 change dimension conv2D
    res = conv_layer(res, dim, (1, 1), (1, 1), name + '_1x1_conv2D')

    # add identity and residue path
    return Add(name=name + '_add')([identity, res])


def resnet(x, num_classes):
    """resnet 4-6-3"""

    # level 0:
    # input: 32X32X3; output: 32X32X64
    x = conv_layer(x, 64, (3, 3), (1, 1), 'level0_conv2D')

    # level 1:
    # input: 32X32X64; output: 32X32X64
    for i in range(num_blocks[0]):
        x = res_block(x, 64, name='level1_block{}'.format(i + 1))

    # level 2:
    # input: 32X32X64; output: 16X16X128
    for i in range(num_blocks[1]):
        x = res_block(x, 128, name='level2_block{}'.format(i + 1))

    # level 3:
    # input: 16X16X128; output: 8X8X256
    for i in range(num_blocks[2]):
        x = res_block(x, 256, name='level3_block{}'.format(i + 1))

    # decode
    x = GlobalAveragePooling2D(name='GAP')(x)
    x = Dense(
        num_classes, activation='softmax',
        kernel_regularizer=regularizers.l2(weight_decay),
        name='FC'
    )(x)
    return x


def resnet_reduced(x, num_classes):
    """resnet 4-6-3, half width"""

    # level 0:
    # input: 32X32X3; output: 32X32X32
    x = conv_layer(x, 32, (3, 3), (1, 1), 'level0_conv2D')

    # level 1:
    # input: 32X32X32; output: 32X32X32
    for i in range(num_blocks[0]):
        x = res_block(x, 32, name='level1_block{}'.format(i + 1))

    # level 2:
    # input: 32X32X32; output: 16X16X64
    for i in range(num_blocks[1]):
        x = res_block(x, 64, name='level2_block{}'.format(i + 1))

    # level 3:
    # input: 16X16X64; output: 8X8X128
    for i in range(num_blocks[2]):
        x = res_block(x, 128, name='level3_block{}'.format(i + 1))

    # decode
    x = GlobalAveragePooling2D(name='GAP')(x)
    x = Dense(
        num_classes, activation='softmax',
        kernel_regularizer=regularizers.l2(weight_decay),
        name='FC'
    )(x)
    return x
