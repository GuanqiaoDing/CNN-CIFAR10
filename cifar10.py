from keras import utils, Model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import numpy as np
import time
import resnet_model

model_name = 'resnet_20'
num_classes = 10
num_blocks = 3  # 3x6+2=20
epochs = 150
batch_size = 128
iterations = 50000 // 128 + 1

# set up learning rate
lr_initial = 0.1
momentum = 0.9


def lr_schedule(epoch):
    if epoch < 80:
        return 0.1
    elif epoch < 120:
        return 0.01
    else:
        return 0.001


# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalize data
channel_mean = np.mean(x_train, axis=(0, 1, 2))
channel_std = np.std(x_train, axis=(0, 1, 2))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

for i in range(3):
    x_train[:, :, :, i] = (x_train[:, :, :, i] - channel_mean[i]) / channel_std[i]
    x_test[:, :, :, i] = (x_test[:, :, :, i] - channel_mean[i]) / channel_std[i]

# labels to categorical
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# build resnext model
img_input = Input(shape=(32, 32, 3), name='input')
img_prediction = resnet_model.resnet(img_input, num_classes, num_blocks)
model = Model(img_input, img_prediction)
print(model.summary())

# model compile
model.compile(
    optimizer=SGD(lr=lr_initial, momentum=momentum, nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# set up callbacks
folder_name = './ckpt/' + model_name + '_' + str(int(time.time()))
cbks = [
    TensorBoard(log_dir='./log/{}'.format(folder_name)),
    LearningRateScheduler(lr_schedule),
    ModelCheckpoint(folder_name + '_{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)
]

# training
history = model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=iterations,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=cbks,
    verbose=1
)

# save model
model.save('{}.h5'.format(model_name))