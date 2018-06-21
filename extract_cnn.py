#!/usr/bin/python
#
from __future__ import print_function
import keras.backend as K
import keras
from keras import applications
from keras import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.optimizers import SGD
#from breakhis_generator_validation import LoadBreakhisList, Generator, GeneratorImgs, ReadImgs, TumorToLabel
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau, TensorBoard, EarlyStopping
import random
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from keras.models import load_model
from keras import regularizers
import sys
import keras.applications.vgg19 as vgg19
import keras.applications.inception_v3 as inception
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import os
#
HEIGHT = 299
WIDTH = 299
#
def ImgToTumor(tumor):
    if(tumor.find("SOB_B_F") != -1):
        return "fibroadenoma"
    if(tumor.find("SOB_M_MC") != -1):
        return "mucinous_carcinoma"
    if(tumor.find("SOB_M_PC") != -1):
        return "papillary_carcinoma"
    if(tumor.find("SOB_M_DC") != -1):
        return "ductal_carcinoma"
    if(tumor.find("SOB_B_TA") != -1):
        return "tubular_adenoma"
    if(tumor.find("SOB_B_A") != -1):
        return "adenosis"
    if(tumor.find("SOB_M_LC") != -1):
        return "lobular_carcinoma"
    if(tumor.find("SOB_B_PT") != -1):
        return "phyllodes_tumor"
    print("Error tumor type: {}".format(tumor))
    exit(0)
#
def build_cnn():

    incep_inst = inception.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=(HEIGHT,WIDTH,3), pooling=None, classes=1000)

    x = incep_inst.output
    #x = Dense(64, activation="relu")(x)
    #x = Dense(2, activation="softmax")(x)
  
    model = Model(inputs=incep_inst.inputs, outputs=x)

    for i in model.layers:
        i.trainable = False

    #trainable_layers = ["block3_pool","block4_conv1","block4_conv2","block4_conv3","block4_conv4","block4_pool","block5_conv1","block5_conv2","block5_conv3","block5_conv4","block5_pool","flatten","fc1","fc2","predictions","dense_1","dense_2"]

    #for i in trainable_layers:
    #    model.get_layer(i).trainable = True

    sgd = SGD(lr=1e-6, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
#
model = build_cnn()
#
#model.summary()
#
#exit(0)
#
func = K.function([model.input] , [model.get_layer("avg_pool").output])
#
with open(sys.argv[1], "r") as f:
    for i in f:
        input_img_name = i[:-1]
        img = image.load_img(input_img_name, target_size=(HEIGHT,WIDTH))
        input_img_data = np.array([img_to_array(img)]).astype('float32')/255
        layer_output = np.array(func([input_img_data])).squeeze()
        img_name = input_img_name.split("/")[-1]
        print("{};{}".format(ImgToTumor(img_name), img_name), end="")
        print(range(0,layer_output.shape[0]))
        exit(0)
        for i in range(0,layer_output.shape[0]):
            print(";{:.7f}".format(layer_output[i]), end="")
        print()
#

