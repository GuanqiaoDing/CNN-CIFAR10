from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Add, Concatenate, \
    Activation, GlobalAveragePooling2D, Lambda
from keras import regularizers

cardinality = 16
bottle_width = 64
chan_per_group = 4  # 64 / 16
num_blocks = 3
momentum = 0.9
epsilon = 1e-5
weight_decay = 5e-4


def conv_bn_relu(x, filters, name, kernel_size=(3, 3), strides=(1, 1)):
    """3X3 common block: conv2D + batch normalization + relu activation"""

    x = Conv2D(
        filters, kernel_size,
        strides=strides, padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=name + '_conv2D'
    )(x)
    x = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_bn')(x)
    x = Activation(activation='relu', name=name + '_relu')(x)
    return x


def conv_bn(x, filters, name):
    """1X1 change dimension block: conv2D + batch normalization"""

    x = Conv2D(
        filters, (1, 1),
        padding='same', use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name=name + '_conv2D'
    )(x)
    x = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_bn')(x)
    return x


def res_block(x, dim, name):
    """residue block: implement diagram c in the original paper"""

    identity = x

    # residual path
    # 1X1 bottleneck
    x = conv_bn_relu(x, bottle_width, kernel_size=(1, 1), name=name + '_bottle')

    # split and 3X3 conv2D
    layers_split = list()
    for i in range(cardinality):
        partial = Lambda(
            lambda y: y[:, :, :, i * chan_per_group: (i + 1) * chan_per_group],
            name=name + '_group{}'.format(i + 1)
        )(x)
        partial = Conv2D(
            chan_per_group, (3, 3),
            padding='same', use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
            name=name + '_group{}_3x3_conv2D'.format(i + 1)
        )(partial)
        layers_split.append(partial)

    # concatenate and restore dimension
    residual = Concatenate(name=name + '_concat')(layers_split)
    residual = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_bn1')(residual)
    residual = Activation(activation='relu', name=name + '_relu1')(residual)
    residual = conv_bn(residual, dim, name=name + '_1x1_conv')

    # add identity and residue path
    added = Add(name=name + '_add')([identity, residual])
    out = Activation(activation='relu', name=name + '_relu2')(added)
    return out


def resnext(x, num_classes):
    """resnext model: mini-version"""

    # level 0:
    # input: 32X32X3; output: 32X32X64
    x = conv_bn_relu(x, 64, name='lv0')

    # level 1:
    # input: 32X32X64; output: 16X16X128
    for i in range(num_blocks):
        x = res_block(x, 64, name='lv1_blk{}'.format(i + 1))
    x = conv_bn_relu(x, 128, strides=(2, 2), name='lv1_DS')

    # level 2:
    # input: 16X16X128; output: 8X8X256
    for i in range(num_blocks):
        x = res_block(x, 128, name='lv2_blk{}'.format(i + 1))
    x = conv_bn_relu(x, 256, strides=(2, 2), name='lv2_DS')

    # level 3:
    # input: 8X8X256; output: 1X1X256
    for i in range(num_blocks):
        x = res_block(x, 256, name='lv3_blk{}'.format(i + 1))
    x = GlobalAveragePooling2D(name='pool')(x)

    # output
    x = Dense(
        num_classes, activation='softmax',
        kernel_regularizer=regularizers.l2(weight_decay),
        name='FC'
    )(x)
    return x
