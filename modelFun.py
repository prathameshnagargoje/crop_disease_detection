import numpy as np
import pandas as pd 
from tensorflow.keras import backend
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential
import cv2


FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
path = 'static/images/'

batch_size = 15


import keras
#from keras.layers.core import Layer
from tensorflow.keras.layers import Layer
import keras.backend as K
import tensorflow as tf

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten

from keras import backend as K 

import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

    
def predict(imgName):
    backend.clear_session()

    input_layer = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=64,
                        filters_3x3_reduce=96,
                        filters_3x3=128,
                        filters_5x5_reduce=16,
                        filters_5x5=32,
                        filters_pool_proj=32,
                        name='inception_3a')

    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=192,
                        filters_5x5_reduce=32,
                        filters_5x5=96,
                        filters_pool_proj=64,
                        name='inception_3b')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=192,
                        filters_3x3_reduce=96,
                        filters_3x3=208,
                        filters_5x5_reduce=16,
                        filters_5x5=48,
                        filters_pool_proj=64,
                        name='inception_4a')


    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(10, activation='softmax', name='auxilliary_output_1')(x1)

    x = inception_module(x,
                        filters_1x1=160,
                        filters_3x3_reduce=112,
                        filters_3x3=224,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4b')

    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=256,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4c')

    x = inception_module(x,
                        filters_1x1=112,
                        filters_3x3_reduce=144,
                        filters_3x3=288,
                        filters_5x5_reduce=32,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4d')


    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(10, activation='softmax', name='auxilliary_output_2')(x2)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_4e')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_5a')

    x = inception_module(x,
                        filters_1x1=384,
                        filters_3x3_reduce=192,
                        filters_3x3=384,
                        filters_5x5_reduce=48,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_5b')

    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

    x = Dropout(0.4)(x)

    x = Dense(10, activation='softmax', name='output')(x)


    model = Model(input_layer, [x, x1, x2], name='inception_v1')

    model.load_weights("model.h5")

    test_filenames = os.listdir(path)
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    test_df = test_df[test_df.filename==imgName]
    nb_samples = test_df.shape[0]

    x_test = []

    x_test.append(cv2.imread(path+imgName))
    
    img_rows = 128
    img_cols = 128
    x_test = np.array([cv2.resize(img, (img_rows,img_cols)) for img in x_test])
    x_test = x_test.astype('float32')
    x_test = x_test / 255.0

    p = model.predict(x_test)

    predict_list = np.argmax(p, axis=-1)

    test_df = pd.DataFrame(predict_list)

    test_df = test_df.replace({9: 'Healthy', 1: 'Bacterial', 3: 'Early', 7: 'Late', 8: 'LeafMold', 4: 'mosaic',
                                         2: 'Septoria', 5: 'Spider', 0: 'Target', 6: 'YellowLeaf'})

    result = test_df.values
    del model
    solution=''

    if result[0]=='Healthy':
        solution=''
        result = 'Healthy'
    elif result[0]=='Bacterial':
        result='Bacterial Virus'
        solution='Bonide liquid copper fungicide'
    elif result[0]=='Early':
        result='Early Blight'
        solution='For best control : Apply copper based fungicides. Repeat every 7-10 days'
    elif result[0]=='Late':
        result='Late Blight'
        solution='Monetory R , All natural disease control'
    elif result[0]=='LeafMold':
        result='Leaf Mold virus'
        solution='apply a fungicide according to the manufacturer’s instructions at the first sign of infection.'
    elif result[0]=='mosaic':
        result='Mosic virus'
        solution='Fungicides will not treat this disease. To prevent, do not use seeds from infected plants.'
    elif result[0]=='Septoria':
        result='Septoria leaf spot'
        solution='Apply bordeaux mixture or ManeborCopper sulphate.'
    elif result[0]=='Spider':
        result='Spider Mites'
        solution='Abameltin, or Bifenazate'
    elif result[0]=='Target':
        result='Target Spot'
        solution='Serenade aarden'
    elif result[0]=='YellowLeaf':
        result='Yellow Leaf curl virus'
        solution='To prevent, Don not over water plants and check soil for lack of Nitrogen.'
    

    return result,solution
