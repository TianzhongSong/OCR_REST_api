# -*- coding:utf-8 -*-
from keras.layers.convolutional import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape,Masking,Lambda,Permute
from keras.layers import Input,Dense,Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import GRU,LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import Adam,SGD,Adadelta
from keras import losses
from keras.layers.wrappers import TimeDistributed

import tensorflow as tf  
import keras.backend.tensorflow_backend as K 


def crnn_model():
    input = Input(shape=(img_h,None,1),name='the_input')
    m = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',name='conv1')(input)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool1')(m)
    m = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same',name='conv2')(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool2')(m)
    m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv3')(m)
    m = BatchNormalization(axis=3)(m)
    m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv4')(m)

    m = ZeroPadding2D(padding=(0,1))(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool3')(m)

    m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv5')(m)
    m = BatchNormalization(axis=3)(m)
    m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv6')(m)

    m = ZeroPadding2D(padding=(0,1))(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool4')(m)
    m = Conv2D(512,kernel_size=(2,2),activation='relu',padding='valid',name='conv7')(m)
    m = BatchNormalization(axis=3)(m)

    m = Permute((2,1,3),name='permute')(m)
    m = TimeDistributed(Flatten(),name='timedistrib')(m)

    m = Bidirectional(GRU(rnnunit,return_sequences=True,implementation=2),name='blstm1')(m)
    m = Dense(rnnunit,name='blstm1_out',activation='linear',)(m)
    m = Bidirectional(GRU(rnnunit,return_sequences=True,implementation=2),name='blstm2')(m)
    y_pred = Dense(nclass,name='blstm2_out',activation='softmax')(m)

    basemodel = Model(inputs=input,outputs=y_pred)
    return basemodel