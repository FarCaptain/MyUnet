
# coding: utf-8

# In[1]:

import matplotlib as mpl
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
import time
#import keras
#from keras.layers.convolutional import Conv2D
import tensorflow as tf
tf.enable_eager_execution()

from tensorflow import keras


print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


# In[2]:

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = keras.layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # second layer
    x = keras.layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    return x

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = False):
    # Contracting Path
    input_img = keras.Input(shape = [224,800,1])#224*800
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = keras.layers.Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = keras.layers.Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = keras.layers.Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = keras.layers.Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

    # Expansive Path
    u6 = keras.layers.Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    u6 = keras.layers.Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

    u7 = keras.layers.Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    u7 = keras.layers.Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    u8 = keras.layers.Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    u8 = keras.layers.Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    u9 = keras.layers.Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = keras.layers.concatenate([u9, c1])
    u9 = keras.layers.Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = keras.Model(inputs=[input_img], outputs=[outputs])
    return model


# In[3]:

def get_number_list(ss):
    ss = ss.split(',')
    ss = [int(x) for x in ss]
    return ss

#224*1016 -> 224*800
def read_file(index):
    mat = np.array([[0]*800 for i in range(224) ])
    string = "D:\\0Sphinx\\SaveData\\data_"+str(index)+".txt"
    f = open(string,"r")
    line = f.readline()
    i = 0
    while line:
        j = 0
        line = get_number_list(line)
        line = line[:800]
        for x in line:
            mat[i][j] = x
            j+=1
        i+=1
        line = f.readline()
    f.close()
    return mat

def read_label(index):
    mat =np.array([[0]*800 for i in range(224) ])
    string = "D:\\0Sphinx\\SaveData\\data_"+str(index)+"_label.txt"
    f = open(string,"r")
    line = f.readline()
    i = 0
    while line:
        j = 0
        line = get_number_list(line)
        line = line[:800]
        for x in line:
            mat[i][j] = x
            j+=1
        i+=1
        line = f.readline()
    f.close()
    return mat

images = []
labels = []
for i in range(1,721):
    images.append(read_file(i))
    string = "D:\\0Sphinx\\SaveData\\data_"+str(i)+"_label.txt"
    if os.path.exists(string):
        labels.append(read_label(i))
    else:
        labels.append(read_label(78)) #零矩阵

X = np.array(images) #数据集
X = np.reshape(X,(-1,224,800,1))
print(X.shape)

Y = np.array(labels)
Y = np.reshape(Y,(-1,224,800,1))
print(Y.shape)

#split into train,valid,test
#x_train_all, x_test, y_train_all, y_test = train_test_split(X,Y,train_size=0.79,random_state=30)


# In[4]:

x_train_all, x_test, y_train_all, y_test = train_test_split(X,Y,train_size=0.896,random_state=30)
print(x_train_all.shape)
print(x_test.shape)


# In[5]:

def make_dataset(images, labels, epochs, batch_size, shuffle = True):
	dataset = tf.data.Dataset.from_tensor_slices((images, labels))
	if shuffle:
		dataset = dataset.shuffle(10000)
	dataset = dataset.repeat(epochs).batch(batch_size)
	return dataset


# In[6]:

x_train, x_valid = x_train_all[:570], x_train_all[570:]
y_train, y_valid = y_train_all[:570], y_train_all[570:]
print(x_train.shape," ", x_valid.shape," ", x_test.shape) #6:2:2
#scale here

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#x_train: [None,224,800,1] -> [None,‭179200,1‬]
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1,1)).reshape(-1,224,800,1)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,224,800,1)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1,1)).reshape(-1,224,800,1)

y_train.astype(np.float32)
y_valid.astype(np.float32)
y_test.astype(np.float32)


# In[7]:

print(np.max(x_train_scaled), np.min(x_train_scaled))


train_set = make_dataset(x_train_scaled,y_train,epochs=10 ,batch_size=10)
valid_set = make_dataset(x_valid_scaled,y_valid,epochs=10 ,batch_size=10)
test_set = make_dataset(x_test_scaled,y_test,epochs=10 ,batch_size=10)


# In[8]:

x_train_scaled= tf.convert_to_tensor(x_train_scaled)
x_valid_scaled= tf.convert_to_tensor(x_valid_scaled)
x_test_scaled= tf.convert_to_tensor(x_test_scaled)

y_train= tf.convert_to_tensor(y_train)
y_valid= tf.convert_to_tensor(y_valid)
y_test= tf.convert_to_tensor(y_test)

#x_train = tf.reshape(x_train,[-1,1,224,1016])
#x_valid = tf.reshape(x_valid,[-1,1,224,1016])
#x_test = tf.reshape(x_test,[-1,224,1016])

print(x_train.shape," ", x_valid.shape," ", x_test.shape) #6:2:2


# In[9]:

from tensorflow.contrib.opt import AdamWOptimizer

def dice_coe(output, target, loss_type='sorensen', axis=(1, 2, 3), smooth=1.):
    """
    Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity of two batch of data,
    usually be used for binary image segmentation
    i.e. labels are binary.
    The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
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

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coe(y_true, y_pred)

from keras import backend as K
def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)

        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed


# adam = tf.train.AdamOptimizer()

# adam with weight decay
adamw = AdamWOptimizer(weight_decay=1e-4)

model = get_unet(x_train_scaled, n_filters = 16, dropout = 0.4, batchnorm = False)
model.compile(optimizer = adamw,
              loss=binary_focal_loss(gamma=2, alpha=0.25),
              metrics=[dice_coe])


# In[10]:

model.summary()

# save the model as file
logdir = './savedModels'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,"unet_focal.h5")

callbacks = [
    tf.keras.callbacks.TensorBoard(logdir),
    tf.keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True, save_weights_only=False),
    #tf.keras.callbacks.EarlyStopping(patience=5, min_delta=2e-4),
]

# In[ ]:

batch_size = 10
history = model.fit(train_set,
                    validation_data=valid_set,
                    shuffle = True,
                    steps_per_epoch  = 570 // batch_size, #batch_size = data_size/steps_per_epoch
                    validation_steps = 1, #75
                    epochs = 10,
                    callbacks = callbacks
                   )  # starts training


# In[ ]:

model.evaluate(test_set, steps = 1)#75


# In[ ]:




# In[ ]:


#result => 70%
