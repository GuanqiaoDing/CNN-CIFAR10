from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Add, Lambda, \
    Activation, GlobalAveragePooling2D, MaxPooling2D, Concatenate
from keras import regularizers, initializers
from keras import backend as K

# according to <arXiv:1512.03385> Table 6.
# Output map size       # layers        # filters
# 32x32                 2n+1            16
# 16x16                 2n              32
# 8x8                   2n              64
#
# Followed by global average pooling and a dense layer with 10 units.
# Total weighted layers: 6n+2
# Total params: 0.27M in resnet-20

momentum = 0.9
epsilon = 1e-5
weight_decay = 1e-4


def conv_layer(x, filters, kernel_size, strides, name):
    """conv2D layer"""
    return Conv2D(
        filters, kernel_size,
        strides=strides, padding='same',
        use_bias=False,
        kernel_initializer=initializers.he_normal(),
        kernel_regularizer=regularizers.l2(weight_decay),
        name=name
    )(x)


def conv_bn_relu(x, filters, kernel_size, strides, name):
    """common conv2D block"""

    x = conv_layer(x, filters, kernel_size, strides, name + '_conv2D')
    x = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_BN')(x)
    x = Activation(activation='relu', name=name + '_relu')(x)
    return x


def conv_bn(x, filters, kernel_size, strides, name):
    """conv2D block without activation"""

    x = conv_layer(x, filters, kernel_size, strides, name + '_conv2D')
    x = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_BN')(x)
    return x


def res_block(x, dim, name):
    """residue block: two 3X3 conv2D stacks"""

    input_dim = int(x.shape[-1])

    # shortcut
    identity = x
    if input_dim != dim:    # option A in the original paper
        identity = MaxPooling2D(
            pool_size=(1, 1), strides=(2, 2),
            padding='same',
            name=name + '_shortcut_pool'
        )(identity)

        identity = Lambda(
            lambda y: K.concatenate([y, K.zeros_like(y)]),
            name=name + '_shortcut_zeropad'
        )(identity)

    # residual path
    res = x
    if input_dim != dim:
        res = conv_bn_relu(res, dim, (3, 3), (2, 2), name + '_res_conv1')
    else:
        res = conv_bn_relu(res, dim, (3, 3), (1, 1), name + '_res_conv1')

    res = conv_bn(res, dim, (3, 3), (1, 1), name + '_res_conv2')

    # add identity and residue path
    out = Add(name=name + '_add')([identity, res])
    out = Activation(activation='relu', name=name + '_relu')(out)
    return out


def resnet(x, num_classes, num_blocks):

    # level 0:
    # input: 32x32x3; output: 32x32x16
    x = conv_bn_relu(x, 16, (3, 3), (1, 1), 'lv0')

    # level 1:
    # input: 32x32x16; output: 32x32x16
    for i in range(num_blocks):
        x = res_block(x, 16, name='lv1_blk{}'.format(i + 1))

    # level 2:
    # input: 32x32x16; output: 16x16x32
    for i in range(num_blocks):
        x = res_block(x, 32, name='lv2_blk{}'.format(i + 1))

    # level 3:
    # input: 16x16x32; output: 8x8x64
    for i in range(num_blocks):
        x = res_block(x, 64, name='lv3_blk{}'.format(i + 1))

    # output
    x = GlobalAveragePooling2D(name='pool')(x)
    x = Dense(
        num_classes, activation='softmax',
        kernel_initializer=initializers.he_normal(),
        kernel_regularizer=regularizers.l2(weight_decay),
        name='FC'
    )(x)
    return x
