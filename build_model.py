from keras import Model
from keras.layers import Input
import resnet_model
import resnext_model

num_classes = 10

# build resnext model
img_input = Input(shape=(32, 32, 3), name='input')
img_prediction = resnext_model.resnext(img_input, num_classes)
resnext = Model(img_input, img_prediction)
print(resnext.summary())



