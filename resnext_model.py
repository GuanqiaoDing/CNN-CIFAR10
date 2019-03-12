from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Add, Concatenate, \
    Activation, GlobalAveragePooling2D, Lambda
from keras import regularizers

cardinality = 8
bottle_width = 64
chan_per_group = 8  # 64 / 8
momentum = 0.9
epsilon = 1e-5
weight_decay = 5e-4


def conv_bn_relu(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    """3X3 common block: conv2D + batch normalization + relu activation"""

    x = Conv2D(
        filters, kernel_size,
        strides=strides, padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = Activation('relu')(x)
    return x


def conv_bn(x, filters):
    """1X1 bottleneck block: conv2D + batch normalization"""

    x = Conv2D(
        filters, (1, 1),
        padding='same', use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    return x


def res_block(x, dim):
    """residue block: implement diagram c in the original paper"""

    if x.shape[-1] != dim:
        identity = conv_bn(x, dim)
    else:
        identity = x

    x = conv_bn_relu(x, bottle_width, kernel_size=(1, 1))

    layers_split = list()
    for i in range(cardinality):
        partial = Lambda(
            lambda y: y[:, :, :, i * chan_per_group: (i + 1) * chan_per_group]
        )(x)
        partial = conv_bn_relu(partial, chan_per_group, kernel_size=(3, 3))
        layers_split.append(partial)

    residual = Concatenate()(layers_split)
    residual = conv_bn(residual, dim)

    added = Add()([identity, residual])
    out = Activation('relu')(added)
    return out


def resnext(x, num_classes):
    """resnext model: mini-version"""

    x = conv_bn_relu(x, 64)

    for _ in range(2):
        x = res_block(x, 64)
    for _ in range(2):
        x = res_block(x, 128)
    for _ in range(2):
        x = res_block(x, 256)

    x = GlobalAveragePooling2D()(x)
    x = Dense(
        num_classes, activation='softmax',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    return x
