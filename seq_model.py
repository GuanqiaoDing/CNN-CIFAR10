from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Activation, GlobalAveragePooling2D, MaxPooling2D
from keras import regularizers

num_classes = 10
momentum = 0.9
epsilon = 1e-5
weight_decay = 5e-4


def conv_bn_relu(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    """conv2D + batch normalization + relu activation"""

    x = Conv2D(
        filters, kernel_size,
        strides=strides, padding='same', use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = BatchNormalization(momentum, epsilon)(x)
    x = Activation('relu')(x)
    return x


def conv_block(x, filters):
    """two conv blocks + 3X3/2 downsampling"""

    x = conv_bn_relu(x, filters)
    x = conv_bn_relu(x, filters)
    x = MaxPooling2D((3, 3), (2, 2), padding='same')(x)
    return x


def seq_model(x, num_classes):
    """sequential model: vgg-like design"""
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x
