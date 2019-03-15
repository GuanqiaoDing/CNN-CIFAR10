from keras import utils, Model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import numpy as np
import math
import time
import seq_model, resnext_model, resnet_model

model_name = 'resnext'
num_classes = 10
epochs = 150
batch_size = 128
iterations = 50000 // 128 + 1

# set up learning rate
lr_initial = 0.001
lr_drop = 0.6
lr_drop_steps = 20


def lr_schedule(epoch):
    return lr_initial * math.pow(lr_drop, math.floor((1 + epoch) / lr_drop_steps))


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
img_input = Input(shape=(32, 32, 3))
img_prediction = resnext_model.resnext(img_input, num_classes)
resnext = Model(img_input, img_prediction)
print(resnext.summary())

# model compile
resnext.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# set up callbacks
folder_name = model_name + '_' + str(int(time.time()))
cbks = [
    TensorBoard(log_dir='./log/{}'.format(folder_name)),
    LearningRateScheduler(lr_schedule),
    ModelCheckpoint(folder_name + '_{epoch:02d}_{val_loss:.2f}.hdf5', save_best_only=True)
]

# training
resnext.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=iterations,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=cbks,
    verbose=1
)

# save model
resnext.save('{}.h5'.format(model_name))