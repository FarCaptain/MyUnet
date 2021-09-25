# coding: utf-8
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import qdarkstyle

import matplotlib as mpl
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
from keras.losses import binary_crossentropy

import tensorflow as tf
tf.enable_eager_execution()

from tensorflow import keras
import keras.backend as K

from sklearn.preprocessing import StandardScaler
from tensorflow.contrib.opt import AdamWOptimizer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # CPU上tf优化

class TrainingPlateform():
    logdir = './savedModels'


    def __init__(self, parent=None):
        #super(TrainingPlateform, self).__init__(parent)
        pass


    def training(self,_nm,_norm,_bn,_bsize,_ep,_loss):

        images = []
        labels = []
        for i in range(1, 21): #721
            images.append(self.read_file(i))
            string = "D:\\0Sphinx\\SaveData\\data_" + str(i) + "_label.txt"
            if os.path.exists(string):
                labels.append(self.read_label(i))
            else:
                labels.append(self.read_label(78))  # 零矩阵

        X = np.array(images)  # 数据集
        X = np.reshape(X, (-1, 224, 800, 1))

        Y = np.array(labels)
        Y = np.reshape(Y, (-1, 224, 800, 1))

        x_train_all, x_test, y_train_all, y_test = train_test_split(X, Y, train_size=0.896, random_state=30)

        x_train, x_valid = x_train_all[:10], x_train_all[10:] #:570 570:
        y_train, y_valid = y_train_all[:10], y_train_all[10:]
        print(x_train.shape, " ", x_valid.shape, " ", x_test.shape)  # 6:2:2

        # scale here
        scaler = StandardScaler()
        # x_train: [None,224,800,1] -> [None,‭179200,1‬]
        x_train_scaled = scaler.fit_transform(
            x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 224, 800, 1)
        x_valid_scaled = scaler.transform(
            x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 224, 800, 1)
        x_test_scaled = scaler.transform(
            x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 224, 800, 1)

        y_train.astype(np.float32)
        y_valid.astype(np.float32)
        y_test.astype(np.float32)

        _x_train = x_train
        _x_valid = x_valid
        _x_test = x_test
        if _norm==True:
            _x_train = x_train_scaled
            _x_valid = x_valid_scaled
            _x_test = x_test_scaled
        train_set = self.make_dataset(_x_train, y_train, _ep, _bsize)
        valid_set = self.make_dataset(_x_valid, y_valid, _ep, _bsize)
        test_set = self.make_dataset(_x_test, y_test, _ep, _bsize)

        x_train_scaled = tf.convert_to_tensor(x_train_scaled)


        # adam with weight decay
        adamw = AdamWOptimizer(weight_decay=1e-4)
        #adamw = tf.keras.optimizers.Adam(lr=1e-4)
        model = self.get_unet(x_train_scaled, n_filters=16, dropout=0.4, batchnorm=_bn) #dropout 也可以调整

        # set loss function
        ls = self.dice_coef_loss
        if _loss==1:
            ls = self.binary_focal_loss(2, 0.25)
        elif _loss==2:
            ls = 'binary_crossentropy'
        elif _loss==3:
            ls = self.dice_p_bce
        else: ls = self.dice_plus_focal
        print(_ep,_bsize,_bn,_norm)
        print(ls)
        model.compile(optimizer=adamw,
                      loss=ls,
                      metrics=[self.dice_coe])

        # In[10]:

        #model.summary()

        # save the model as file
        logdir = './savedModels'
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        output_model_file = os.path.join(logdir, _nm+".h5")

        callbacks = [
            #tf.keras.callbacks.TensorBoard(logdir),
            tf.keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True, save_weights_only=False),
            # tf.keras.callbacks.EarlyStopping(patience=5, min_delta=2e-4),
        ]
        print("Here?")



        batch_size = _bsize
        self.history = model.fit(train_set,
                            validation_data=valid_set,
                            shuffle=True,
                            steps_per_epoch=10 // batch_size,  # batch_size = data_size/steps_per_epoch
                            validation_steps=1,  # 75
                            epochs=_ep,
                            callbacks=callbacks
                            )  # starts training


        model.evaluate(test_set, steps=1)  # 75

        #print(QMessageBox.information(self, "提醒", "训练完成！", QMessageBox.Yes, QMessageBox.Yes))


    def showLoss(self):
        # 绘制训练 & 验证的损失值
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def showAcc(self):
        # 绘制训练 & 验证的准确率值
        plt.plot(self.history.history['dice_coe'])
        plt.plot(self.history.history['val_dice_coe'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


    def conv2d_block(self,input_tensor, n_filters, kernel_size=3, batchnorm=True):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                                kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        # second layer
        x = keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                                kernel_initializer='he_normal', padding='same')(x)
        if batchnorm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        return x

    def get_unet(self,input_img, n_filters=16, dropout=0.1, batchnorm=False):
        # Contracting Path
        input_img = keras.Input(shape=[224, 800, 1])  # 224*800
        c1 = self.conv2d_block(input_img, n_filters * 1, 3, batchnorm)
        p1 = keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = keras.layers.Dropout(dropout)(p1)

        c2 = self.conv2d_block(p1, n_filters * 2, 3, batchnorm)
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = keras.layers.Dropout(dropout)(p2)

        c3 = self.conv2d_block(p2, n_filters * 4, 3, batchnorm)
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)
        p3 = keras.layers.Dropout(dropout)(p3)

        c4 = self.conv2d_block(p3, n_filters * 8, 3, batchnorm)
        p4 = keras.layers.MaxPooling2D((2, 2))(c4)
        p4 = keras.layers.Dropout(dropout)(p4)

        c5 = self.conv2d_block(p4, n_filters * 16, 3, batchnorm)

        # Expansive Path
        u6 = keras.layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = keras.layers.concatenate([u6, c4])
        u6 = keras.layers.Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters * 8, 3, batchnorm)

        u7 = keras.layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = keras.layers.concatenate([u7, c3])
        u7 = keras.layers.Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters * 4, 3, batchnorm)

        u8 = keras.layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = keras.layers.concatenate([u8, c2])
        u8 = keras.layers.Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters * 2, 3, batchnorm)

        u9 = keras.layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = keras.layers.concatenate([u9, c1])
        u9 = keras.layers.Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters * 1, 3, batchnorm)

        outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = keras.Model(inputs=[input_img], outputs=[outputs])
        return model

    # In[3]:

    def get_number_list(self,ss):
        ss = ss.split(',')
        ss = [int(x) for x in ss]
        return ss

    # 224*1016 -> 224*800
    def read_file(self,index):
        mat = np.array([[0] * 800 for i in range(224)])
        string = ".\\SaveData\\data_" + str(index) + ".txt"
        f = open(string, "r")
        line = f.readline()
        i = 0
        while line:
            j = 0
            line = self.get_number_list(line)
            line = line[:800]
            for x in line:
                mat[i][j] = x
                j += 1
            i += 1
            line = f.readline()
        f.close()
        return mat

    def read_label(self,index):
        mat = np.array([[0] * 800 for i in range(224)])
        string = ".\\SaveData\\data_" + str(index) + "_label.txt"
        f = open(string, "r")
        line = f.readline()
        i = 0
        while line:
            j = 0
            line = self.get_number_list(line)
            line = line[:800]
            for x in line:
                mat[i][j] = x
                j += 1
            i += 1
            line = f.readline()
        f.close()
        return mat


    def make_dataset(self, images, labels, epochs, batch_size, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        if shuffle:
            dataset = dataset.shuffle(10000)
        dataset = dataset.repeat(epochs).batch(batch_size)
        return dataset


    def dice_coe(self,output, target, loss_type='sorensen', axis=(1, 2, 3), smooth=1.):
        inse = tf.reduce_sum(output * target, axis=axis)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output, axis=axis)
            r = tf.reduce_sum(target * target, axis=axis)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output, axis=axis)
            r = tf.reduce_sum(target, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)
        return dice

    def dice_coef_loss(self, y_true, y_pred):
        return 1. - self.dice_coe(y_true, y_pred)

    def binary_focal_loss(self,gamma=2, alpha=0.25):

        alpha = tf.constant(alpha, dtype=tf.float32)
        gamma = tf.constant(gamma, dtype=tf.float32)

        def binary_focal_loss_fixed(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

            p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
            focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
            return K.mean(focal_loss)

        return binary_focal_loss_fixed

    def dice_p_bce(self, in_gt, in_pred):
        return 1.0 * tf.reduce_mean(binary_crossentropy(in_gt, in_pred)) + self.dice_coef_loss(in_gt, in_pred)



    def binary_focal_loss_fixed(self,y_true, y_pred):
        gamma = 2
        alpha = 0.25
        l = 0.1
        alpha = tf.constant(alpha, dtype=tf.float32)
        gamma = tf.constant(gamma, dtype=tf.float32)

        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return l * K.mean(focal_loss)

    def dice_plus_focal(self, y_true, y_pred):
        loss_1 = self.dice_coef_loss(y_true, y_pred)
        loss_2 = self.binary_focal_loss_fixed(y_true, y_pred)
        return loss_1 + loss_2
