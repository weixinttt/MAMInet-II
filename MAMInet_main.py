# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:03:55 2024

@author: weixxy
"""

import time
import keras.callbacks as Callback
import h5py
import math
import pylab
import keras
import numpy as np
import scipy.io as sio
import scipy.io as scio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.layers import Activation,Input,Dense,Lambda,Conv2D,Concatenate,Permute,concatenate,AveragePooling2D,BatchNormalization,Reshape,Multiply,Add,Conv3D,Flatten,Dropout,GlobalAveragePooling2D
iterations=200

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
tf.config.experimental_run_functions_eagerly(True)

dataset=0

if dataset==0:
    data=h5py.File('Italy_plus.mat', 'r')
    batchsize=256
    source='MUUFLresult2/'
if dataset==1:
    data=h5py.File('Italy_plus.mat', 'r')
    source='result2/'
    batchsize=256

X_spatial_all=data['HSI'][()].transpose(3,2,1,0)
LiDAR_all=data['LiDAR'][()].transpose(2,1,0)
LiDAR_all=np.expand_dims(LiDAR_all,-1)
act_Y_train_all=data['act_Y_train'][()].transpose(1,0)
indexi_all=data['indexi'][()].transpose(1,0)
indexj_all=data['indexj'][()].transpose(1,0)

X_spatial_all=X_spatial_all.astype('float32')
LiDAR_all=LiDAR_all.astype('float32')
act_Y_train_all=act_Y_train_all.astype('int')
indexi_all=indexi_all.astype('float32')
indexj_all=indexj_all.astype('float32')

act_Y_train_all[act_Y_train_all==-1]=0

slide_size=9
if slide_size==3:
    X_spatial_all=X_spatial_all[:,4:7,4:7,:]
    LiDAR_all=LiDAR_all[:,4:7,4:7,:]
if slide_size==5:
    X_spatial_all=X_spatial_all[:,3:8,3:8,:]
    LiDAR_all=LiDAR_all[:,3:8,3:8,:]
if slide_size==7:
    X_spatial_all=X_spatial_all[:,2:9,2:9,:]
    LiDAR_all=LiDAR_all[:,2:9,2:9,:]
if slide_size==9:
    X_spatial_all=X_spatial_all[:,1:10,1:10,:]
    LiDAR_all=LiDAR_all[:,1:10,1:10,:]
if slide_size==11:
    X_spatial_all=X_spatial_all[:,:,:,:]
    LiDAR_all=LiDAR_all[:,:,:,:]
###############################################################################
scaler=MinMaxScaler(feature_range=(0,1))

X_spatial_all_=X_spatial_all.reshape([X_spatial_all.shape[0],X_spatial_all.shape[1]*X_spatial_all.shape[2]*X_spatial_all.shape[3]])
LiDAR_all_=LiDAR_all.reshape([LiDAR_all.shape[0],LiDAR_all.shape[1]*LiDAR_all.shape[2]*LiDAR_all.shape[3]])

X_spatial_all_=scaler.fit_transform(X_spatial_all_)
LiDAR_all_=scaler.fit_transform(LiDAR_all_)

X_spatial_all=X_spatial_all_.reshape([X_spatial_all.shape[0],X_spatial_all.shape[1],X_spatial_all.shape[2],X_spatial_all.shape[3]])
LiDAR_all=LiDAR_all_.reshape([LiDAR_all.shape[0],LiDAR_all.shape[1],LiDAR_all.shape[2],LiDAR_all.shape[3]])


act_Y_train_all=np.reshape(act_Y_train_all,act_Y_train_all.shape[0])
indexi_all=np.reshape(indexi_all,indexi_all.shape[0])
indexj_all=np.reshape(indexj_all,indexj_all.shape[0])

act_Y_train=act_Y_train_all

randpaixv=act_Y_train_all.argsort()
X_spatial_all=X_spatial_all[randpaixv]
LiDAR_all=LiDAR_all[randpaixv]
indexi_all=indexi_all[randpaixv]
indexj_all=indexj_all[randpaixv]
act_Y_train_all=act_Y_train_all[randpaixv]

X_spatial_all=X_spatial_all[act_Y_train_all>0]
LiDAR_all=LiDAR_all[act_Y_train_all>0]
indexi_all=indexi_all[act_Y_train_all>0]
indexj_all=indexj_all[act_Y_train_all>0]
act_Y_train_all=act_Y_train_all[act_Y_train_all>0]

indices=np.arange(X_spatial_all.shape[0])
indices_train,indices_test,act_Y_train_train,act_Y_train_test=train_test_split(indices,act_Y_train_all,test_size=0.99,stratify=act_Y_train_all)

X_spatial_train=X_spatial_all[indices_train,:,:]
LiDAR_train=LiDAR_all[indices_train,:,:]
act_Y_train_train=act_Y_train_all[indices_train]
indexi_train=indexi_all[indices_train]
indexj_train=indexj_all[indices_train]

X_spatial_test=X_spatial_all[indices_test,:,:]
LiDAR_test=LiDAR_all[indices_test,:,:]
act_Y_train_test=act_Y_train_all[indices_test]
indexi_test=indexi_all[indices_test]
indexj_test=indexj_all[indices_test]

act_Y_train_train=np_utils.to_categorical(act_Y_train_train-1)
act_Y_train_test=np_utils.to_categorical(act_Y_train_test-1)
#################################
def sampling(args):
    z_mean,z_log_var=args
    epsilon=K.random_normal(shape=K.shape(z_mean))
    return z_mean+K.exp(z_log_var/2)*epsilon
class EpochCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()
        
    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime
        
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        self.sum_time = sum(self.times)


#################################
activation='tanh'
kernel_regularizer=tf.keras.regularizers.l2(0.01)
lr=0.0005
kl_balance=0.01



H1_input=Input(shape=(X_spatial_train.shape[1],X_spatial_train.shape[2],X_spatial_train.shape[3]))
H2_input=Input(shape=(X_spatial_train.shape[1],X_spatial_train.shape[2],X_spatial_train.shape[3]))

L1_input=Input(shape=(LiDAR_train.shape[1],LiDAR_train.shape[2],1))
L2_input=Input(shape=(LiDAR_train.shape[1],LiDAR_train.shape[2],1))

indexi1=Input(shape=(1,))
indexi2=Input(shape=(1,))

indexj1=Input(shape=(1,))
indexj2=Input(shape=(1,))

Hx1_1_one=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H1_input)
Hx1_1_one=BatchNormalization()(Hx1_1_one)
Hx1_1_one=Activation(activation)(Hx1_1_one)

Hx2_1_one=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H2_input)
Hx2_1_one=BatchNormalization()(Hx2_1_one)
Hx2_1_one=Activation(activation)(Hx2_1_one)

Rx1_1_one=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx1_1_one)
Rx1_1_one=BatchNormalization()(Rx1_1_one)
Rx1_1_one=Activation(activation)(Rx1_1_one)

Rx2_1_one=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx2_1_one)
Rx2_1_one=BatchNormalization()(Rx2_1_one)
Rx2_1_one=Activation(activation)(Rx2_1_one)

Sx1_1_one=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx1_1_one)
Sx1_1_one=BatchNormalization()(Sx1_1_one)
Sx1_1_one=Activation(activation)(Sx1_1_one)

Sx2_1_one=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx2_1_one)
Sx2_1_one=BatchNormalization()(Sx2_1_one)
Sx2_1_one=Activation(activation)(Sx2_1_one)

Lx1_1_one=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L1_input)
Lx1_1_one=BatchNormalization()(Lx1_1_one)
Lx1_1_one=Activation(activation)(Lx1_1_one)

Lx2_1_one=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L2_input)
Lx2_1_one=BatchNormalization()(Lx2_1_one)
Lx2_1_one=Activation(activation)(Lx2_1_one)

Lx1_2_one=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_1_one)
Lx1_2_one=BatchNormalization()(Lx1_2_one)
Lx1_2_one=Activation(activation)(Lx1_2_one)

Lx2_2_one=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_1_one)
Lx2_2_one=BatchNormalization()(Lx2_2_one)
Lx2_2_one=Activation(activation)(Lx2_2_one)

Hx1_middle=GlobalAveragePooling2D()(H1_input)
Hx2_middle=GlobalAveragePooling2D()(H2_input)
Lx1_middle=GlobalAveragePooling2D()(L1_input)
Lx2_middle=GlobalAveragePooling2D()(L2_input)
Rx1_2_mean=GlobalAveragePooling2D()(Rx1_1_one)
Rx2_2_mean=GlobalAveragePooling2D()(Rx2_1_one)
Sx1_2_mean=GlobalAveragePooling2D()(Sx1_1_one)
Sx2_2_mean=GlobalAveragePooling2D()(Sx2_1_one)
##############################################################################
Lx1_middle=Lambda(lambda x:K.tile(x,[1,int(X_spatial_train.shape[3])]))(Lx1_middle)
Lx2_middle=Lambda(lambda x:K.tile(x,[1,int(X_spatial_train.shape[3])]))(Lx2_middle)

HSI_gradient=Lambda(lambda x:1/(K.abs(x[0]-x[1])*K.sqrt(K.square(x[2]-x[3])+K.square(x[4]-x[5])+1e-9)))([Hx1_middle,Hx2_middle,indexi1,indexi2,indexj1,indexj2])
LiDAR_gradient=Lambda(lambda x:1/(K.abs(x[0]-x[1])*K.sqrt(K.square(x[2]-x[3])+K.square(x[4]-x[5])+1e-9)))([Lx1_middle,Lx2_middle,indexi1,indexi2,indexj1,indexj2])

HSI_gradient=Activation('sigmoid')(HSI_gradient)
LiDAR_gradient=Activation('sigmoid')(LiDAR_gradient)
##################9x9编码器Rx1-Lx1##################################
Rx1_h=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_1_one)
Rx1_z_mean=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_h)
Rx1_z_log_var=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_h)

Lx1_h=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_2_one)
Lx1_z_mean=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_h)
Lx1_z_log_var=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_h)

Rx1_z=Lambda(sampling,output_shape=(X_spatial_all.shape[-1],))([Rx1_z_mean,Lx1_z_log_var])
Lx1_z=Lambda(sampling,output_shape=(X_spatial_all.shape[-1],))([Lx1_z_mean,Rx1_z_log_var])
##################9x9解码器Rx1###################################
Rx1_h_decoded=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_z)
Rx1_decoded_mean=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_h_decoded)
##############################################################
Rx1_x_loss=Lambda(lambda x:K.sum((x[0]-x[1]),axis=-1))([H1_input,Rx1_decoded_mean])
Rx1_kl_loss=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1))([Rx1_z_mean,Rx1_z_log_var])
Rx1_vae_loss=Lambda(lambda x:K.mean(x[0]+kl_balance*x[1]),output_shape=[1,])([Rx1_x_loss,Rx1_kl_loss])
##################9x9解码器Lx1###################################
Lx1_h_decoded=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_z)
Lx1_decoded_mean=Conv2D(LiDAR_train.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_h_decoded)  
##############################################################
Lx1_x_loss=Lambda(lambda x:K.sum((x[0]-x[1]),axis=-1))([L1_input,Lx1_decoded_mean])
Lx1_kl_loss=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1))([Lx1_z_mean,Lx1_z_log_var])
Lx1_vae_loss=Lambda(lambda x:K.mean(x[0]+kl_balance*x[1]),output_shape=[1,])([Lx1_x_loss,Lx1_kl_loss])
##################9x9编码器Rx2-Lx2##################################
Rx2_h=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_1_one)
Rx2_z_mean=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_h)
Rx2_z_log_var=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_h)

Lx2_h=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_2_one)
Lx2_z_mean=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_h)
Lx2_z_log_var=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_h)

Rx2_z=Lambda(sampling,output_shape=(X_spatial_all.shape[-1],))([Rx2_z_mean,Lx2_z_log_var])
Lx2_z=Lambda(sampling,output_shape=(X_spatial_all.shape[-1],))([Lx2_z_mean,Rx2_z_log_var])
##################9x9解码器Rx2###################################
Rx2_h_decoded=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_z)
Rx2_decoded_mean=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_h_decoded)
##############################################################
Rx2_x_loss=Lambda(lambda x:K.sum((x[0]-x[1]),axis=-1))([H2_input,Rx2_decoded_mean])
Rx2_kl_loss=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1))([Rx2_z_mean,Rx2_z_log_var])
Rx2_vae_loss=Lambda(lambda x:K.mean(x[0]+kl_balance*x[1]),output_shape=[1,])([Rx2_x_loss,Rx2_kl_loss])
##################9x9解码器Lx2###################################
Lx2_h_decoded=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_z)
Lx2_decoded_mean=Conv2D(LiDAR_train.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_h_decoded)  
##############################################################
Lx2_x_loss=Lambda(lambda x:K.sum((x[0]-x[1]),axis=-1))([L2_input,Lx2_decoded_mean])
Lx2_kl_loss=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1))([Lx2_z_mean,Lx2_z_log_var])
Lx2_vae_loss=Lambda(lambda x:K.mean(x[0]+kl_balance*x[1]),output_shape=[1,])([Lx2_x_loss,Lx2_kl_loss])
##############################################################



##############################################################
mid1=Multiply()([Rx1_2_mean,Sx1_2_mean])
mid2=Multiply()([Rx2_2_mean,Sx2_2_mean])





loss1=Lambda(lambda x:K.mean(K.square(x[0]-x[1]),axis=-1),name='loss1')([Hx1_middle,mid1])
loss2=Lambda(lambda x:K.mean(K.square(x[0]-x[1]),axis=-1),name='loss2')([Hx2_middle,mid2])

loss3=Lambda(lambda x:K.mean(x[2]*K.square(x[0]-x[1]),axis=-1),name='loss3')([Rx1_2_mean,Rx2_2_mean,HSI_gradient])
loss4=Lambda(lambda x:K.mean(x[2]*K.square(x[0]-x[1]),axis=-1),name='loss4')([Sx1_2_mean,Sx2_2_mean,LiDAR_gradient])




loss_RS=Lambda(lambda x:x[0]+x[1]+x[2]+x[3],output_shape=[1,])([loss1,loss2,loss3,loss4])

loss_VAE=Lambda(lambda x:x[0]+x[1]+x[2]+x[3],output_shape=[1,])([Rx1_vae_loss,Rx2_vae_loss,Lx1_vae_loss,Lx2_vae_loss])



###############################################################################



X_spatial_all_2=data['HSI'][()].transpose(3,2,1,0)
LiDAR_all_2=data['LiDAR'][()].transpose(2,1,0)
LiDAR_all_2=np.expand_dims(LiDAR_all_2,-1)
act_Y_train_all_2=data['act_Y_train'][()].transpose(1,0)
indexi_all_2=data['indexi'][()].transpose(1,0)
indexj_all_2=data['indexj'][()].transpose(1,0)

X_spatial_all_2=X_spatial_all_2.astype('float32')
LiDAR_all_2=LiDAR_all_2.astype('float32')
act_Y_train_all_2=act_Y_train_all_2.astype('int')
indexi_all_2=indexi_all_2.astype('float32')
indexj_all_2=indexj_all_2.astype('float32')

act_Y_train_all_2[act_Y_train_all_2==-1]=0




slide_size_2=5
if slide_size_2==3:
    X_spatial_all_2=X_spatial_all_2[:,4:7,4:7,:]
    LiDAR_all_2=LiDAR_all_2[:,4:7,4:7,:]
if slide_size_2==5:
    X_spatial_all_2=X_spatial_all_2[:,3:8,3:8,:]
    LiDAR_all_2=LiDAR_all_2[:,3:8,3:8,:]
if slide_size_2==7:
    X_spatial_all_2=X_spatial_all_2[:,2:9,2:9,:]
    LiDAR_all_2=LiDAR_all_2[:,2:9,2:9,:]
if slide_size_2==9:
    X_spatial_all_2=X_spatial_all_2[:,1:10,1:10,:]
    LiDAR_all_2=LiDAR_all_2[:,1:10,1:10,:]
if slide_size_2==11:
    X_spatial_all_2=X_spatial_all_2[:,:,:,:]
    LiDAR_all_2=LiDAR_all_2[:,:,:,:]
###############################################################################


X_spatial_all_2_=X_spatial_all_2.reshape([X_spatial_all_2.shape[0],X_spatial_all_2.shape[1]*X_spatial_all_2.shape[2]*X_spatial_all_2.shape[3]])
LiDAR_all_2_=LiDAR_all_2.reshape([LiDAR_all_2.shape[0],LiDAR_all_2.shape[1]*LiDAR_all_2.shape[2]*LiDAR_all_2.shape[3]])

X_spatial_all_2_=scaler.fit_transform(X_spatial_all_2_)
LiDAR_all_2_=scaler.fit_transform(LiDAR_all_2_)

X_spatial_all_2=X_spatial_all_2_.reshape([X_spatial_all_2.shape[0],X_spatial_all_2.shape[1],X_spatial_all_2.shape[2],X_spatial_all_2.shape[3]])
LiDAR_all_2=LiDAR_all_2_.reshape([LiDAR_all_2.shape[0],LiDAR_all_2.shape[1],LiDAR_all_2.shape[2],LiDAR_all_2.shape[3]])


act_Y_train_all_2=np.reshape(act_Y_train_all_2,act_Y_train_all_2.shape[0])
indexi_all_2=np.reshape(indexi_all_2,indexi_all_2.shape[0])
indexj_all_2=np.reshape(indexj_all_2,indexj_all_2.shape[0])

act_Y_train_2=act_Y_train_all_2


X_spatial_all_2=X_spatial_all_2[randpaixv]
LiDAR_all_2=LiDAR_all_2[randpaixv]
indexi_all_2=indexi_all_2[randpaixv]
indexj_all_2=indexj_all_2[randpaixv]
act_Y_train_all_2=act_Y_train_all_2[randpaixv]

X_spatial_all_2=X_spatial_all_2[act_Y_train_all_2>0]
LiDAR_all_2=LiDAR_all_2[act_Y_train_all_2>0]
indexi_all_2=indexi_all_2[act_Y_train_all_2>0]
indexj_all_2=indexj_all_2[act_Y_train_all_2>0]
act_Y_train_all_2=act_Y_train_all_2[act_Y_train_all_2>0]

indices_2=np.arange(X_spatial_all.shape[0])
indices_train_2,indices_test_2,act_Y_train_train_2,act_Y_train_test_2=train_test_split(indices_2,act_Y_train_all_2,test_size=0.99,stratify=act_Y_train_all_2)

X_spatial_train_2=X_spatial_all_2[indices_train_2,:,:]
LiDAR_train_2=LiDAR_all_2[indices_train_2,:,:]
act_Y_train_train_2=act_Y_train_all_2[indices_train_2]
indexi_train_2=indexi_all_2[indices_train_2]
indexj_train_2=indexj_all_2[indices_train_2]

X_spatial_test_2=X_spatial_all_2[indices_test_2,:,:]
LiDAR_test_2=LiDAR_all_2[indices_test_2,:,:]
act_Y_train_test_2=act_Y_train_all_2[indices_test_2]
indexi_test_2=indexi_all_2[indices_test_2]
indexj_test_2=indexj_all_2[indices_test_2]

act_Y_train_train_2=np_utils.to_categorical(act_Y_train_train_2-1)
act_Y_train_test_2=np_utils.to_categorical(act_Y_train_test_2-1)



H3_input=Input(shape=(X_spatial_train_2.shape[1],X_spatial_train_2.shape[2],X_spatial_train_2.shape[3]))
H4_input=Input(shape=(X_spatial_train_2.shape[1],X_spatial_train_2.shape[2],X_spatial_train_2.shape[3]))

L3_input=Input(shape=(LiDAR_train_2.shape[1],LiDAR_train_2.shape[2],1))
L4_input=Input(shape=(LiDAR_train_2.shape[1],LiDAR_train_2.shape[2],1))

indexi3=Input(shape=(1,))
indexi4=Input(shape=(1,))

indexj3=Input(shape=(1,))
indexj4=Input(shape=(1,))

Hx1_1_two=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H3_input)
Hx1_1_two=BatchNormalization()(Hx1_1_two)
Hx1_1_two=Activation(activation)(Hx1_1_two)

Hx2_1_two=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H4_input)
Hx2_1_two=BatchNormalization()(Hx2_1_two)
Hx2_1_two=Activation(activation)(Hx2_1_two)

Rx1_1_two=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx1_1_two)
Rx1_1_two=BatchNormalization()(Rx1_1_two)
Rx1_1_two=Activation(activation)(Rx1_1_two)

Rx2_1_two=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx2_1_two)
Rx2_1_two=BatchNormalization()(Rx2_1_two)
Rx2_1_two=Activation(activation)(Rx2_1_two)

Sx1_1_two=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx1_1_two)
Sx1_1_two=BatchNormalization()(Sx1_1_two)
Sx1_1_two=Activation(activation)(Sx1_1_two)

Sx2_1_two=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx2_1_two)
Sx2_1_two=BatchNormalization()(Sx2_1_two)
Sx2_1_two=Activation(activation)(Sx2_1_two)

Lx1_1_two=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L3_input)
Lx1_1_two=BatchNormalization()(Lx1_1_two)
Lx1_1_two=Activation(activation)(Lx1_1_two)

Lx2_1_two=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L4_input)
Lx2_1_two=BatchNormalization()(Lx2_1_two)
Lx2_1_two=Activation(activation)(Lx2_1_two)

Lx1_2_two=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_1_two)
Lx1_2_two=BatchNormalization()(Lx1_2_two)
Lx1_2_two=Activation(activation)(Lx1_2_two)

Lx2_2_two=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_1_two)
Lx2_2_two=BatchNormalization()(Lx2_2_two)
Lx2_2_two=Activation(activation)(Lx2_2_two)

Hx3_middle=GlobalAveragePooling2D()(H3_input)
Hx4_middle=GlobalAveragePooling2D()(H4_input)
Lx3_middle=GlobalAveragePooling2D()(L3_input)
Lx4_middle=GlobalAveragePooling2D()(L4_input)
Rx1_2_mean_2=GlobalAveragePooling2D()(Rx1_1_two)
Rx2_2_mean_2=GlobalAveragePooling2D()(Rx2_1_two)
Sx1_2_mean_2=GlobalAveragePooling2D()(Sx1_1_two)
Sx2_2_mean_2=GlobalAveragePooling2D()(Sx2_1_two)
##############################################################################
Lx3_middle=Lambda(lambda x:K.tile(x,[1,int(X_spatial_train_2.shape[3])]))(Lx3_middle)
Lx4_middle=Lambda(lambda x:K.tile(x,[1,int(X_spatial_train_2.shape[3])]))(Lx4_middle)

HSI_gradient_2=Lambda(lambda x:1/(K.abs(x[0]-x[1])*K.sqrt(K.square(x[2]-x[3])+K.square(x[4]-x[5])+1e-9)))([Hx3_middle,Hx4_middle,indexi3,indexi4,indexj3,indexj4])
LiDAR_gradient_2=Lambda(lambda x:1/(K.abs(x[0]-x[1])*K.sqrt(K.square(x[2]-x[3])+K.square(x[4]-x[5])+1e-9)))([Lx3_middle,Lx4_middle,indexi3,indexi4,indexj3,indexj4])

HSI_gradient_2=Activation('sigmoid')(HSI_gradient_2)
LiDAR_gradient_2=Activation('sigmoid')(LiDAR_gradient_2)
##################7x7编码器Rx1-Lx1##################################
Rx1_h_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_1_two)
Rx1_z_mean_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_h_2)
Rx1_z_log_var_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_h_2)

Lx1_h_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_2_two)
Lx1_z_mean_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_h_2)
Lx1_z_log_var_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_h_2)

Rx1_z_2=Lambda(sampling,output_shape=(X_spatial_all_2.shape[-1],))([Rx1_z_mean_2,Lx1_z_log_var_2])
Lx1_z_2=Lambda(sampling,output_shape=(X_spatial_all_2.shape[-1],))([Lx1_z_mean_2,Rx1_z_log_var_2])
##################7x7解码器Rx1###################################
Rx1_h_decoded_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_z_2)
Rx1_decoded_mean_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_h_decoded_2)
##############################################################
Rx1_x_loss_2=Lambda(lambda x:K.sum((x[0]-x[1]),axis=-1))([H3_input,Rx1_decoded_mean_2])
Rx1_kl_loss_2=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1))([Rx1_z_mean_2,Rx1_z_log_var_2])
Rx1_vae_loss_2=Lambda(lambda x:K.mean(x[0]+kl_balance*x[1]),output_shape=[1,])([Rx1_x_loss_2,Rx1_kl_loss_2])
##################7x7解码器Lx1###################################
Lx1_h_decoded_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_z_2)
Lx1_decoded_mean_2=Conv2D(LiDAR_train_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_h_decoded_2)  
##############################################################
Lx1_x_loss_2=Lambda(lambda x:K.sum((x[0]-x[1]),axis=-1))([L3_input,Lx1_decoded_mean_2])
Lx1_kl_loss_2=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1))([Lx1_z_mean_2,Lx1_z_log_var_2])
Lx1_vae_loss_2=Lambda(lambda x:K.mean(x[0]+kl_balance*x[1]),output_shape=[1,])([Lx1_x_loss_2,Lx1_kl_loss_2])
##################7x7编码器Rx2-Lx2##################################
Rx2_h_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_1_two)
Rx2_z_mean_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_h_2)
Rx2_z_log_var_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_h_2)

Lx2_h_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_2_two)
Lx2_z_mean_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_h_2)
Lx2_z_log_var_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_h_2)

Rx2_z_2=Lambda(sampling,output_shape=(X_spatial_all_2.shape[-1],))([Rx2_z_mean_2,Lx2_z_log_var_2])
Lx2_z_2=Lambda(sampling,output_shape=(X_spatial_all_2.shape[-1],))([Lx2_z_mean_2,Rx2_z_log_var_2])
##################7x7解码器Rx2###################################
Rx2_h_decoded_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_z_2)
Rx2_decoded_mean_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_h_decoded_2)
##############################################################
Rx2_x_loss_2=Lambda(lambda x:K.sum((x[0]-x[1]),axis=-1))([H4_input,Rx2_decoded_mean_2])
Rx2_kl_loss_2=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1))([Rx2_z_mean_2,Rx2_z_log_var_2])
Rx2_vae_loss_2=Lambda(lambda x:K.mean(x[0]+kl_balance*x[1]),output_shape=[1,])([Rx2_x_loss_2,Rx2_kl_loss_2])
##################7x7解码器Lx2###################################
Lx2_h_decoded_2=Conv2D(X_spatial_all_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_z_2)
Lx2_decoded_mean_2=Conv2D(LiDAR_train_2.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_h_decoded_2)  
##############################################################
Lx2_x_loss_2=Lambda(lambda x:K.sum((x[0]-x[1]),axis=-1))([L4_input,Lx2_decoded_mean_2])
Lx2_kl_loss_2=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1))([Lx2_z_mean_2,Lx2_z_log_var_2])
Lx2_vae_loss_2=Lambda(lambda x:K.mean(x[0]+kl_balance*x[1]),output_shape=[1,])([Lx2_x_loss_2,Lx2_kl_loss_2])
##############################################################






mid1_2=Multiply()([Rx1_2_mean_2,Sx1_2_mean_2])
mid2_2=Multiply()([Rx2_2_mean_2,Sx2_2_mean_2])

loss1_2=Lambda(lambda x:K.mean(K.square(x[0]-x[1]),axis=-1),name='loss1_2')([Hx3_middle,mid1_2])
loss2_2=Lambda(lambda x:K.mean(K.square(x[0]-x[1]),axis=-1),name='loss2_2')([Hx4_middle,mid2_2])

loss3_2=Lambda(lambda x:K.mean(x[2]*K.square(x[0]-x[1]),axis=-1),name='loss3_2')([Rx1_2_mean_2,Rx2_2_mean_2,HSI_gradient_2])
loss4_2=Lambda(lambda x:K.mean(x[2]*K.square(x[0]-x[1]),axis=-1),name='loss4_2')([Sx1_2_mean_2,Sx2_2_mean_2,LiDAR_gradient_2])




loss_RS_2=Lambda(lambda x:x[0]+x[1]+x[2]+x[3],output_shape=[1,])([loss1_2,loss2_2,loss3_2,loss4_2])

loss_VAE_2=Lambda(lambda x:x[0]+x[1]+x[2]+x[3],output_shape=[1,])([Rx1_vae_loss_2,Rx2_vae_loss_2,Lx1_vae_loss_2,Lx2_vae_loss_2])







X_spatial_all_3=data['HSI'][()].transpose(3,2,1,0)
LiDAR_all_3=data['LiDAR'][()].transpose(2,1,0)
LiDAR_all_3=np.expand_dims(LiDAR_all_3,-1)
act_Y_train_all_3=data['act_Y_train'][()].transpose(1,0)
indexi_all_3=data['indexi'][()].transpose(1,0)
indexj_all_3=data['indexj'][()].transpose(1,0)

X_spatial_all_3=X_spatial_all_3.astype('float32')
LiDAR_all_3=LiDAR_all_3.astype('float32')
act_Y_train_all_3=act_Y_train_all_3.astype('int')
indexi_all_3=indexi_all_3.astype('float32')
indexj_all_3=indexj_all_3.astype('float32')

act_Y_train_all_3[act_Y_train_all_3==-1]=0




slide_size_3=3
if slide_size_3==3:
    X_spatial_all_3=X_spatial_all_3[:,4:7,4:7,:]
    LiDAR_all_3=LiDAR_all_3[:,4:7,4:7,:]
if slide_size_3==5:
    X_spatial_all_3=X_spatial_all_3[:,3:8,3:8,:]
    LiDAR_all_3=LiDAR_all_3[:,3:8,3:8,:]
if slide_size_3==7:
    X_spatial_all_3=X_spatial_all_3[:,2:9,2:9,:]
    LiDAR_all_3=LiDAR_all_3[:,2:9,2:9,:]
if slide_size_3==9:
    X_spatial_all_3=X_spatial_all_3[:,1:10,1:10,:]
    LiDAR_all_3=LiDAR_all_3[:,1:10,1:10,:]
if slide_size_3==11:
    X_spatial_all_3=X_spatial_all_3[:,:,:,:]
    LiDAR_all_3=LiDAR_all_3[:,:,:,:]
###############################################################################


X_spatial_all_3_=X_spatial_all_3.reshape([X_spatial_all_3.shape[0],X_spatial_all_3.shape[1]*X_spatial_all_3.shape[2]*X_spatial_all_3.shape[3]])
LiDAR_all_3_=LiDAR_all_3.reshape([LiDAR_all_3.shape[0],LiDAR_all_3.shape[1]*LiDAR_all_3.shape[2]*LiDAR_all_3.shape[3]])

X_spatial_all_3_=scaler.fit_transform(X_spatial_all_3_)
LiDAR_all_3_=scaler.fit_transform(LiDAR_all_3_)

X_spatial_all_3=X_spatial_all_3_.reshape([X_spatial_all_3.shape[0],X_spatial_all_3.shape[1],X_spatial_all_3.shape[2],X_spatial_all_3.shape[3]])
LiDAR_all_3=LiDAR_all_3_.reshape([LiDAR_all_3.shape[0],LiDAR_all_3.shape[1],LiDAR_all_3.shape[2],LiDAR_all_3.shape[3]])



act_Y_train_all_3=np.reshape(act_Y_train_all_3,act_Y_train_all_3.shape[0])
indexi_all_3=np.reshape(indexi_all_3,indexi_all_3.shape[0])
indexj_all_3=np.reshape(indexj_all_3,indexj_all_3.shape[0])

act_Y_train_3=act_Y_train_all_3


X_spatial_all_3=X_spatial_all_3[randpaixv]
LiDAR_all_3=LiDAR_all_3[randpaixv]
indexi_all_3=indexi_all_3[randpaixv]
indexj_all_3=indexj_all_3[randpaixv]
act_Y_train_all_3=act_Y_train_all_3[randpaixv]

X_spatial_all_3=X_spatial_all_3[act_Y_train_all_3>0]
LiDAR_all_3=LiDAR_all_3[act_Y_train_all_3>0]
indexi_all_3=indexi_all_3[act_Y_train_all_3>0]
indexj_all_3=indexj_all_3[act_Y_train_all_3>0]
act_Y_train_all_3=act_Y_train_all_3[act_Y_train_all_3>0]

indices_3=np.arange(X_spatial_all.shape[0])
indices_train_3,indices_test_3,act_Y_train_train_3,act_Y_train_test_3=train_test_split(indices_3,act_Y_train_all_3,test_size=0.99,stratify=act_Y_train_all_3)

X_spatial_train_3=X_spatial_all_3[indices_train_3,:,:]
LiDAR_train_3=LiDAR_all_3[indices_train_3,:,:]
act_Y_train_train_3=act_Y_train_all_3[indices_train_3]
indexi_train_3=indexi_all_3[indices_train_3]
indexj_train_3=indexj_all_3[indices_train_3]

X_spatial_test_3=X_spatial_all_3[indices_test_3,:,:]
LiDAR_test_3=LiDAR_all_3[indices_test_3,:,:]
act_Y_train_test_3=act_Y_train_all_3[indices_test_3]
indexi_test_3=indexi_all_3[indices_test_3]
indexj_test_3=indexj_all_3[indices_test_3]

act_Y_train_train_3=np_utils.to_categorical(act_Y_train_train_3-1)
act_Y_train_test_3=np_utils.to_categorical(act_Y_train_test_3-1)



H5_input=Input(shape=(X_spatial_train_3.shape[1],X_spatial_train_3.shape[2],X_spatial_train_3.shape[3]))
H6_input=Input(shape=(X_spatial_train_3.shape[1],X_spatial_train_3.shape[2],X_spatial_train_3.shape[3]))

L5_input=Input(shape=(LiDAR_train_3.shape[1],LiDAR_train_3.shape[2],1))
L6_input=Input(shape=(LiDAR_train_3.shape[1],LiDAR_train_3.shape[2],1))

indexi5=Input(shape=(1,))
indexi6=Input(shape=(1,))

indexj5=Input(shape=(1,))
indexj6=Input(shape=(1,))

Hx1_1_three=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H5_input)
Hx1_1_three=BatchNormalization()(Hx1_1_three)
Hx1_1_three=Activation(activation)(Hx1_1_three)

Hx2_1_three=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H6_input)
Hx2_1_three=BatchNormalization()(Hx2_1_three)
Hx2_1_three=Activation(activation)(Hx2_1_three)

Rx1_1_three=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx1_1_three)
Rx1_1_three=BatchNormalization()(Rx1_1_three)
Rx1_1_three=Activation(activation)(Rx1_1_three)

Rx2_1_three=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx2_1_three)
Rx2_1_three=BatchNormalization()(Rx2_1_three)
Rx2_1_three=Activation(activation)(Rx2_1_three)

Sx1_1_three=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx1_1_three)
Sx1_1_three=BatchNormalization()(Sx1_1_three)
Sx1_1_three=Activation(activation)(Sx1_1_three)

Sx2_1_three=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx2_1_three)
Sx2_1_three=BatchNormalization()(Sx2_1_three)
Sx2_1_three=Activation(activation)(Sx2_1_three)

Lx1_1_three=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L5_input)
Lx1_1_three=BatchNormalization()(Lx1_1_three)
Lx1_1_three=Activation(activation)(Lx1_1_three)

Lx2_1_three=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L6_input)
Lx2_1_three=BatchNormalization()(Lx2_1_three)
Lx2_1_three=Activation(activation)(Lx2_1_three)

Lx1_2_three=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_1_three)
Lx1_2_three=BatchNormalization()(Lx1_2_three)
Lx1_2_three=Activation(activation)(Lx1_2_three)

Lx2_2_three=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_1_three)
Lx2_2_three=BatchNormalization()(Lx2_2_three)
Lx2_2_three=Activation(activation)(Lx2_2_three)

Hx5_middle=GlobalAveragePooling2D()(H5_input)
Hx6_middle=GlobalAveragePooling2D()(H6_input)
Lx5_middle=GlobalAveragePooling2D()(L5_input)
Lx6_middle=GlobalAveragePooling2D()(L6_input)
Rx1_2_mean_3=GlobalAveragePooling2D()(Rx1_1_three)
Rx2_2_mean_3=GlobalAveragePooling2D()(Rx2_1_three)
Sx1_2_mean_3=GlobalAveragePooling2D()(Sx1_1_three)
Sx2_2_mean_3=GlobalAveragePooling2D()(Sx2_1_three)
##############################################################################
Lx5_middle=Lambda(lambda x:K.tile(x,[1,int(X_spatial_train_3.shape[3])]))(Lx5_middle)
Lx6_middle=Lambda(lambda x:K.tile(x,[1,int(X_spatial_train_3.shape[3])]))(Lx6_middle)

HSI_gradient_3=Lambda(lambda x:1/(K.abs(x[0]-x[1])*K.sqrt(K.square(x[2]-x[3])+K.square(x[4]-x[5])+1e-9)))([Hx5_middle,Hx6_middle,indexi5,indexi6,indexj5,indexj6])
LiDAR_gradient_3=Lambda(lambda x:1/(K.abs(x[0]-x[1])*K.sqrt(K.square(x[2]-x[3])+K.square(x[4]-x[5])+1e-9)))([Lx5_middle,Lx6_middle,indexi5,indexi6,indexj5,indexj6])

HSI_gradient_3=Activation('sigmoid')(HSI_gradient_3)
LiDAR_gradient_3=Activation('sigmoid')(LiDAR_gradient_3)
##################5x5编码器Rx1-Lx1##################################
Rx1_h_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_1_three)
Rx1_z_mean_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_h_3)
Rx1_z_log_var_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_h_3)

Lx1_h_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_2_three)
Lx1_z_mean_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_h_3)
Lx1_z_log_var_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_h_3)

Rx1_z_3=Lambda(sampling,output_shape=(X_spatial_all_3.shape[-1],))([Rx1_z_mean_3,Lx1_z_log_var_3])
Lx1_z_3=Lambda(sampling,output_shape=(X_spatial_all_3.shape[-1],))([Lx1_z_mean_3,Rx1_z_log_var_3])
##################5x5解码器Rx1###################################
Rx1_h_decoded_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_z_3)
Rx1_decoded_mean_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx1_h_decoded_3)
##############################################################
Rx1_x_loss_3=Lambda(lambda x:K.sum((x[0]-x[1]),axis=-1))([H5_input,Rx1_decoded_mean_3])
Rx1_kl_loss_3=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1))([Rx1_z_mean_3,Rx1_z_log_var_3])
Rx1_vae_loss_3=Lambda(lambda x:K.mean(x[0]+kl_balance*x[1]),output_shape=[1,])([Rx1_x_loss_3,Rx1_kl_loss_3])
##################5x5解码器Lx1###################################
Lx1_h_decoded_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_z_3)
Lx1_decoded_mean_3=Conv2D(LiDAR_train_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1_h_decoded_3)  
##############################################################
Lx1_x_loss_3=Lambda(lambda x:K.sum((x[0]-x[1]),axis=-1))([L5_input,Lx1_decoded_mean_3])
Lx1_kl_loss_3=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1))([Lx1_z_mean_3,Lx1_z_log_var_3])
Lx1_vae_loss_3=Lambda(lambda x:K.mean(x[0]+kl_balance*x[1]),output_shape=[1,])([Lx1_x_loss_3,Lx1_kl_loss_3])
##################5x5编码器Rx2-Lx2##################################
Rx2_h_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_1_three)
Rx2_z_mean_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_h_3)
Rx2_z_log_var_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_h_3)

Lx2_h_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_2_three)
Lx2_z_mean_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_h_3)
Lx2_z_log_var_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_h_3)

Rx2_z_3=Lambda(sampling,output_shape=(X_spatial_all_3.shape[-1],))([Rx2_z_mean_3,Lx2_z_log_var_3])
Lx2_z_3=Lambda(sampling,output_shape=(X_spatial_all_3.shape[-1],))([Lx2_z_mean_3,Rx2_z_log_var_3])
##################5x5解码器Rx2###################################
Rx2_h_decoded_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_z_3)
Rx2_decoded_mean_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Rx2_h_decoded_3)
##############################################################
Rx2_x_loss_3=Lambda(lambda x:K.sum((x[0]-x[1]),axis=-1))([H6_input,Rx2_decoded_mean_3])
Rx2_kl_loss_3=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1))([Rx2_z_mean_3,Rx2_z_log_var_3])
Rx2_vae_loss_3=Lambda(lambda x:K.mean(x[0]+kl_balance*x[1]),output_shape=[1,])([Rx2_x_loss_3,Rx2_kl_loss_3])
##################5x5解码器Lx2###################################
Lx2_h_decoded_3=Conv2D(X_spatial_all_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_z_3)
Lx2_decoded_mean_3=Conv2D(LiDAR_train_3.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2_h_decoded_3)  
##############################################################
Lx2_x_loss_3=Lambda(lambda x:K.sum((x[0]-x[1]),axis=-1))([L6_input,Lx2_decoded_mean_3])
Lx2_kl_loss_3=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1))([Lx2_z_mean_3,Lx2_z_log_var_3])
Lx2_vae_loss_3=Lambda(lambda x:K.mean(x[0]+kl_balance*x[1]),output_shape=[1,])([Lx2_x_loss_3,Lx2_kl_loss_3])
##############################################################


mid1_3=Multiply()([Rx1_2_mean_3,Sx1_2_mean_3])
mid2_3=Multiply()([Rx2_2_mean_3,Sx2_2_mean_3])




loss1_3=Lambda(lambda x:K.mean(K.square(x[0]-x[1]),axis=-1),name='loss1_3')([Hx5_middle,mid1_3])
loss2_3=Lambda(lambda x:K.mean(K.square(x[0]-x[1]),axis=-1),name='loss2_3')([Hx6_middle,mid2_3])

loss3_3=Lambda(lambda x:K.mean(x[2]*K.square(x[0]-x[1]),axis=-1),name='loss3_3')([Rx1_2_mean_3,Rx2_2_mean_3,HSI_gradient_3])
loss4_3=Lambda(lambda x:K.mean(x[2]*K.square(x[0]-x[1]),axis=-1),name='loss4_3')([Sx1_2_mean_3,Sx2_2_mean_3,LiDAR_gradient_3])



loss_RS_3=Lambda(lambda x:x[0]+x[1]+x[2]+x[3],output_shape=[1,])([loss1_3,loss2_3,loss3_3,loss4_3])

loss_VAE_3=Lambda(lambda x:x[0]+x[1]+x[2]+x[3],output_shape=[1,])([Rx1_vae_loss_3,Rx2_vae_loss_3,Lx1_vae_loss_3,Lx2_vae_loss_3])



loss_R_S = Lambda(lambda x:x[0]+x[1]+x[2]+x[3],output_shape=[1,])([loss_RS,loss_RS_2,loss_RS_3])
loss_V = Lambda(lambda x:x[0]+x[1]+x[2],output_shape=[1,])([loss_VAE,loss_VAE_2,loss_VAE_3])


Rx1_z_mean_=GlobalAveragePooling2D()(Rx1_z_mean)
Lx1_z_mean_=GlobalAveragePooling2D()(Lx1_z_mean)

Rx2_z_mean_=GlobalAveragePooling2D()(Rx2_z_mean)
Lx2_z_mean_=GlobalAveragePooling2D()(Lx2_z_mean)



Rx1_z_mean_2_=GlobalAveragePooling2D()(Rx1_z_mean_2)
Lx1_z_mean_2_=GlobalAveragePooling2D()(Lx1_z_mean_2)

Rx2_z_mean_2_=GlobalAveragePooling2D()(Rx2_z_mean_2)
Lx2_z_mean_2_=GlobalAveragePooling2D()(Lx2_z_mean_2)

Rx1_z_mean_3_=GlobalAveragePooling2D()(Rx1_z_mean_3)
Lx1_z_mean_3_=GlobalAveragePooling2D()(Lx1_z_mean_3)

Rx2_z_mean_3_=GlobalAveragePooling2D()(Rx2_z_mean_3)
Lx2_z_mean_3_=GlobalAveragePooling2D()(Lx2_z_mean_3)



Rx_Lx_out=Concatenate()([Rx1_z_mean_,Lx1_z_mean_,Rx1_z_mean_2_,Lx1_z_mean_2_,Rx1_z_mean_3_,Lx1_z_mean_3_])
# Rx_Lx_out=GlobalAveragePooling2D()(Rx_Lx_out)

Rx_Lx_out_=Dense(act_Y_train_train.shape[1],activation='softmax')(Rx_Lx_out)




network=keras.models.Model([H1_input,H2_input,H3_input,H4_input,H5_input,H6_input,L1_input,L2_input,L3_input,L4_input,L5_input,L6_input,indexi1,indexi2,indexi3,indexi4,indexi5,indexi6,indexj1,indexj2,indexj3,indexj4,indexj5,indexj6],
                           [Rx_Lx_out_,loss_V,loss_R_S])
network.compile(loss=['categorical_crossentropy','mean_squared_error','mean_squared_error'],
                loss_weights=[1,0.01,1],
                optimizer=keras.optimizers.Adam(lr=lr,decay=0.01))
network.summary()
##############################################################
maxacc=0
maxkappa=0
maxp=0
maxr=0
maxf1_score=0
epoch_time_callback = EpochCallback()

for iter in range(iterations):
    index1=[ind for ind in range(int(X_spatial_train.shape[0]))]
    np.random.shuffle(index1)
    X_spatial_train1=X_spatial_train[index1]
    LiDAR_train1=LiDAR_train[index1]
    act_Y_train_train1=act_Y_train_train[index1]
    indexi_train1=indexi_train[index1]
    indexj_train1=indexj_train[index1]   
    
    index2=[ind for ind in range(int(X_spatial_train.shape[0]))]
    np.random.shuffle(index2)
    X_spatial_train2=X_spatial_train[index2]
    LiDAR_train2=LiDAR_train[index2]
    act_Y_train_train2=act_Y_train_train[index2]
    indexi_train2=indexi_train[index2]
    indexj_train2=indexj_train[index2]
    
    index3=[ind for ind in range(int(X_spatial_train_2.shape[0]))]
    np.random.shuffle(index3)
    X_spatial_train1_2=X_spatial_train_2[index3]
    LiDAR_train1_2=LiDAR_train_2[index3]
    act_Y_train_train1_2=act_Y_train_train_2[index3]
    indexi_train1_2=indexi_train_2[index3]
    indexj_train1_2=indexj_train_2[index3]   
    
    index4=[ind for ind in range(int(X_spatial_train_2.shape[0]))]
    np.random.shuffle(index4)
    X_spatial_train2_2=X_spatial_train_2[index4]
    LiDAR_train2_2=LiDAR_train_2[index4]
    act_Y_train_train2_2=act_Y_train_train_2[index4]
    indexi_train2_2=indexi_train_2[index4]
    indexj_train2_2=indexj_train_2[index4]
    
    index5=[ind for ind in range(int(X_spatial_train_3.shape[0]))]
    np.random.shuffle(index5)
    X_spatial_train1_3=X_spatial_train_3[index5]
    LiDAR_train1_3=LiDAR_train_3[index5]
    act_Y_train_train1_3=act_Y_train_train_3[index5]
    indexi_train1_3=indexi_train_3[index5]
    indexj_train1_3=indexj_train_3[index5]   
    
    index6=[ind for ind in range(int(X_spatial_train_3.shape[0]))]
    np.random.shuffle(index6)
    X_spatial_train2_3=X_spatial_train_3[index6]
    LiDAR_train2_3=LiDAR_train_3[index6]
    act_Y_train_train2_3=act_Y_train_train_3[index6]
    indexi_train2_3=indexi_train_3[index6]
    indexj_train2_3=indexj_train_3[index6]
    
    
    
    history=network.fit([X_spatial_train1,
                         X_spatial_train2,
                         X_spatial_train1_2,
                         X_spatial_train2_2,
                         X_spatial_train1_3,
                         X_spatial_train2_3,
                         LiDAR_train1,
                         LiDAR_train2,
                         LiDAR_train1_2,
                         LiDAR_train2_2,
                         LiDAR_train1_3,
                         LiDAR_train2_3,
                         indexi_train1,
                         indexi_train2,
                         indexi_train1_2,
                         indexi_train2_2,
                         indexi_train1_3,
                         indexi_train2_3,
                         indexj_train1,
                         indexj_train2,
                         indexj_train1_2,
                         indexj_train2_2,
                         indexj_train1_3,
                         indexj_train2_3],
                        [act_Y_train_train1,
                         np.zeros([act_Y_train_train.shape[0],1]),
                         np.zeros([act_Y_train_train.shape[0],1])],
                        batch_size=batchsize,
                        epochs=100,
                        shuffle=True,
                        verbose=1,
                        callbacks=[epoch_time_callback])
    Test_loss=network.predict([X_spatial_test,X_spatial_test,X_spatial_test_2,X_spatial_test_2,X_spatial_test_3,X_spatial_test_3,LiDAR_test,LiDAR_test,LiDAR_test_2,LiDAR_test_2,LiDAR_test_3,LiDAR_test_3,indexi_test,indexi_test,indexi_test_2,indexi_test_2,indexi_test_3,indexi_test_3,indexj_test,indexj_test,indexj_test_2,indexj_test_2,indexj_test_3,indexj_test_3])
    predicted=Test_loss[0]
    
    
    predicted_label=predicted.argmax(axis=1)
    raw_label=act_Y_train_test.argmax(axis=1)

    acc=accuracy_score(predicted_label,raw_label)
    
    if acc>maxacc:
        acc=acc
        Pred_result_ = predicted_label
        generator_component=keras.models.Model([H1_input,H2_input,L1_input,L2_input,indexi1,indexi2,indexj1,indexj2],[Rx1_1_one,Rx2_1_one,Sx1_1_one,Sx2_1_one])
        [R1_result,R2_result,S1_result,S2_result]=generator_component.predict([X_spatial_all,X_spatial_all,LiDAR_all,LiDAR_all,indexi_all,indexi_all,indexj_all,indexj_all])
        R=np.zeros([int(indexi_all.max()),int(indexj_all.max()),X_spatial_all.shape[-1]])
        for iii in range(indexi_all.shape[0]):
            R[int(indexi_all[iii]-1),int(indexj_all[iii]-1),:]=R1_result[iii,int((R1_result.shape[1]-1)/2),int((R1_result.shape[1]-1)/2),:]
        S=np.zeros([int(indexi_all.max()),int(indexj_all.max()),X_spatial_all.shape[-1]])
        for iii in range(indexi_all.shape[0]):
            S[int(indexi_all[iii]-1),int(indexj_all[iii]-1),:]=S1_result[iii,int((S1_result.shape[1]-1)/2),int((S1_result.shape[1]-1)/2),:]            

        
        maxacc=acc
        maxOA=maxacc
        maxAllAcc=recall_score(raw_label,predicted_label,average=None)
        maxAA=recall_score(raw_label,predicted_label,average='macro')
        maxkappa=cohen_kappa_score(np.array(predicted_label).reshape(-1,1),np.array(raw_label).reshape(-1,1))
        maxp=precision_score(raw_label,predicted_label,average='macro')
        maxf1score=f1_score(raw_label,predicted_label,average='macro')
        

        MAP=np.zeros([int(indexi_all.max()),int(indexj_all.max())])
        for ii in range(act_Y_train_test.shape[0]):
            MAP[int(indexi_test[ii]-1),int(indexj_test[ii]-1)]=predicted.argmax(axis=1)[ii]+1
        for ii in range(act_Y_train_train.shape[0]):
            MAP[int(indexi_train[ii]-1),int(indexj_train[ii]-1)]=(act_Y_train_train.argmax(axis=1))[ii]+1
        

        name=source+'net_result_'+str(iter)+'_'+str(maxacc)+'.mat'
        sio.savemat(name, {'Pred_result': predicted_label,
                             'raw_label':raw_label,
                             'maxacc':maxacc,
                             'maxAllAcc':maxAllAcc,
                             'maxOA':maxOA,
                             'maxAA':maxAA,
                             'maxkappa':maxkappa,
                             'maxp':maxp,
                             'maxf1score':maxf1score,
                              'R':R,
                              'S':S,
                             'MAP':MAP,
                             'total_time':epoch_time_callback.sum_time
                             })
