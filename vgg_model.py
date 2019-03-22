from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Activation, GlobalAveragePooling2D, MaxPooling2D
from keras import regularizers, initializers

# total params: 0.27M in vgg-20

momentum = 0.9
epsilon = 1e-5
weight_decay = 1e-4


def conv_bn_relu(x, filters, name, kernel_size=(3, 3), strides=(1, 1)):
    """conv2D + batch normalization + relu activation"""

    x = Conv2D(
        filters, kernel_size,
        strides=strides, padding='same', use_bias=False,
        kernel_initializer=initializers.he_normal(),
        kernel_regularizer=regularizers.l2(weight_decay),
        name=name + '_conv2D'
    )(x)
    x = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_BN')(x)
    x = Activation('relu', name=name + '_relu')(x)
    return x


def conv_blocks(x, filters, num_blocks, name):
    """two conv, downsampling if dimension not match"""

    for i in range(num_blocks):
        if int(x.shape[-1]) != filters:
            x = conv_bn_relu(x, filters, strides=(2, 2), name=name + '_blk{}_conv1'.format(i + 1))
        else:
            x = conv_bn_relu(x, filters, name + '_blk{}_conv1'.format(i + 1))
        x = conv_bn_relu(x, filters, name + '_blk{}_conv2'.format(i + 1))
    return x


def vgg_model(x, num_classes, num_blocks):
    """sequential model without shortcut, same number of parameters as its resnet counterpart"""

    # level 0:
    # input: 32x32x3; output: 32x32x16
    x = conv_bn_relu(x, 16, name='lv0')

    # level 1:
    # input: 32x32x16; output: 32x32x16
    x = conv_blocks(x, 16, num_blocks, name='lv1')

    # level 2:
    # input: 32x32x16; output: 16x16x32
    x = conv_blocks(x, 32, num_blocks, name='lv2')

    # level 3:
    # input: 16x16x32; output: 8x8x64
    x = conv_blocks(x, 64, num_blocks, name='lv3')

    # output
    x = GlobalAveragePooling2D(name='global_pool')(x)
    x = Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=initializers.he_normal(),
        kernel_regularizer=regularizers.l2(weight_decay),
        name='FC'
    )(x)
    return x
