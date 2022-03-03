from __future__ import absolute_import

weightspath = "thyroid_att_v1.h5"

from layer_utils import *
from activations import GELU, Snake

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from numpy.random import seed
seed(99)
import tensorflow as tf
tf.random.set_seed(99) 
from tensorflow.python.client import device_lib 
import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import skimage.morphology as morphology

def dice_coef(y_true,y_pred, loss_type='jaccard', smooth=1.):

    y_true_f = tf.reshape(y_true,[-1])
    y_pred_f = tf.reshape(y_pred,[-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true,y_pred, loss_type='jaccard', smooth=1.):

    y_true_f = tf.cast(tf.reshape(y_true,[-1]),tf.float32)
    y_pred_f =tf.cast(tf.reshape(y_pred,[-1]),tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (1-(2. * intersection + smooth) / (union + smooth))


# In[ ]:

def ResUNET_a_block(X, channel, kernel_size=3, dilation_num=1.0, activation='ReLU', batch_norm=False, name='res_a_block'):
    
    X_res = []
    
    for i, d in enumerate(dilation_num):
        
        X_res.append(CONV_stack(X, channel, kernel_size=kernel_size, stack_num=2, dilation_rate=d, 
                                activation=activation, batch_norm=batch_norm, name='{}_stack{}'.format(name, i)))
        
    if len(X_res) > 1:
        return add(X_res)
    
    else:
        return X_res[0]


def ResUNET_a_right(X, X_list, channel, kernel_size=3, dilation_num=[1,], 
                    activation='ReLU', unpool=True, batch_norm=False, name='right0'):
 
    pool_size = 2
    
    X = decode_layer(X, channel, pool_size, unpool, 
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))
        
    # <--- *stacked convolutional can be applied here
    X = concatenate([X,]+X_list, axis=3, name=name+'_concat')
    
    # Stacked convolutions after concatenation 
    X = ResUNET_a_block(X, channel, kernel_size=kernel_size, dilation_num=dilation_num, activation=activation, 
                        batch_norm=batch_norm, name='{}_resblock'.format(name))
     
    return X

def resunet_a_2d_base(input_tensor, filter_num, dilation_num,
                      aspp_num_down=256, aspp_num_up=128, activation='ReLU',
                      batch_norm=True, pool=True, unpool=True, name='resunet'):

    pool_size = 2
    
    activation_func = eval(activation)
    
    depth_ = len(filter_num)
    X_skip = []
    
    # ----- #
    # rejecting auto-mode from this base function
    if isinstance(dilation_num[0], int):
        raise ValueError('`resunet_a_2d_base` does not support automated determination of `dilation_num`.')
    else:
        dilation_ = dilation_num
    # ----- #
    
    X = input_tensor
    
    # input mapping with 1-by-1 conv
    X = Conv2D(filter_num[0], 1, 1, dilation_rate=1, padding='same', 
               use_bias=True, name='{}_input_mapping'.format(name))(X)
    X = activation_func(name='{}_input_activation'.format(name))(X)
    X_skip.append(X)
    # ----- #
    
    X = ResUNET_a_block(X, filter_num[0], kernel_size=3, dilation_num=dilation_[0], 
                        activation=activation, batch_norm=batch_norm, name='{}_res0'.format(name)) 
    X_skip.append(X)

    for i, f in enumerate(filter_num[1:]):
        ind_ = i+1
        
        X = encode_layer(X, f, pool_size, pool, activation=activation, 
                         batch_norm=batch_norm, name='{}_down{}'.format(name, i))
        X = ResUNET_a_block(X, f, kernel_size=3, dilation_num=dilation_[ind_], activation=activation, 
                            batch_norm=batch_norm, name='{}_resblock_{}'.format(name, ind_))
        X_skip.append(X)

    X = ASPP_conv(X, aspp_num_down, activation=activation, batch_norm=batch_norm, name='{}_aspp_bottom'.format(name))

    X_skip = X_skip[:-1][::-1]
    dilation_ = dilation_[:-1][::-1]
    
    for i, f in enumerate(filter_num[:-1][::-1]):

        X = ResUNET_a_right(X, [X_skip[i],], f, kernel_size=3, activation=activation, dilation_num=dilation_[i], 
                            unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))

    X = concatenate([X_skip[-1], X], name='{}_concat_out'.format(name))

    X = ASPP_conv(X, aspp_num_up, activation=activation, batch_norm=batch_norm, name='{}_aspp_out'.format(name))
    
    return X


def resunet_a_2d(input_size, filter_num, n_labels, dilation_num=[1, 3, 15, 31],
                 aspp_num_down=256, aspp_num_up=128, activation='ReLU', output_activation='Sigmoid',#'Softmax', 
                 batch_norm=True, pool=True, unpool=True, name='resunet'):

    
    activation_func = eval(activation)
    depth_ = len(filter_num)
    
    X_skip = []
    
    # input_size cannot have None
    if input_size[0] is None or input_size[1] is None:
        raise ValueError('`resunet_a_2d` does not support NoneType input shape')
        
    # ----- #
    if isinstance(dilation_num[0], int):
        print("Received dilation rates: {}".format(dilation_num))
    
        deep_ = (depth_-2)//2
        dilation_ = [[] for _ in range(depth_)]
        
        print("Received dilation rates are not defined on a per downsampling level basis.")
        print("Automated determinations are applied with the following details:")
        
        for i in range(depth_):
            if i <= 1:
                dilation_[i] += dilation_num
            elif i > 1 and i <= deep_+1:
                dilation_[i] += dilation_num[:-1]
            else:
                dilation_[i] += [1,]
            print('\tdepth-{}, dilation_rate = {}'.format(i, dilation_[i]))
            
    else:
        dilation_ = dilation_num
    # ----- #
    
    IN = Input(input_size)
    
    # base
    X = resunet_a_2d_base(IN, filter_num, dilation_,
                          aspp_num_down=aspp_num_down, aspp_num_up=aspp_num_up, activation=activation,
                          batch_norm=batch_norm, pool=pool, unpool=unpool, name=name)
    
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))

    model = Model([IN], [OUT,], name='{}_model'.format(name))
    
    return model


# In[ ]:

##PRE-PROCESSING INCASE OF RAW DATA
def crop_image(img):##Crops meaningful region
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img1 = img1>0
    img1 = ndimage.morphology.binary_closing(img1,structure=np.ones((5,5)))
    img1 = morphology.binary_opening(img1,np.ones((120,120)))
    contours,_ = cv2.findContours(img1.astype(np.uint8), cv2.RETR_EXTERNAL, 2)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    img = img[y:y+h,x:x+w,:]
    return img,[x,y,w,h]

def process(image,b=0): ##Detects if multiple images are present in a single image
    if b==0:
        kern_size = 3
        can = cv2.medianBlur(image, kern_size)
        threshold_lower = 30
        threshold_upper = 220
        can = cv2.Canny(can, threshold_lower, threshold_upper)
        #can = cv2.Canny(image,50,200,apertureSize = 3)
    else:
        can = cv2.Canny(image,100,150,apertureSize = 3)
    imm = np.zeros_like(image)
    minLineLength = 0
    maxLineGap = 300
    lines = cv2.HoughLinesP(can,rho=1,theta=np.pi/180,threshold=100,minLineLength=minLineLength,maxLineGap=maxLineGap)
    if lines is not None:
        for x1,y1,x2,y2 in lines[0]:
            if not ((x1>= imm.shape[1] - 1) or (x1<=1)):
                imm = cv2.line(imm,(x1,y1),(x2,y2),255,1)
        flag = False
        for i in range(imm.shape[1]):
            count = np.count_nonzero(imm[:,i])
            if count>(2*(imm.shape[0]/3)):
                col = i
                flag = True
                break
        if flag==False:
            return False,0
        else:
            return True,col
    elif b==0:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)
        return process(image,1)
    else:
        return False,0
    
def preprocess(img,dim): ##main pre-processing function to be called, returns processed image and dimensions for post-processing
    img,origcrop = crop_image(img)
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (croprow,cropcol) = img1.shape
    image = img1[:,int(img1.shape[1]/2)-30:int(img1.shape[1]/2)+30]
    flag,col = process(image)
    #flag=False
    if flag==False:
        img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
        return origcrop , [croprow,cropcol,0] , [img]
    else:
        col = col + int(img1.shape[1]/2)-30
        return origcrop , [croprow,cropcol,col] , [cv2.resize(img[:,:col], dim, interpolation = cv2.INTER_CUBIC),cv2.resize(img[:,col:], dim, interpolation = cv2.INTER_CUBIC)]




from tensorflow.keras.optimizers import *
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model_a = resunet_a_2d(input_size = (256,256,3),filter_num=[32,64,128,256,512], n_labels=1)
model_a.load_weights(weightspath)


# In[ ]:


def get_prediction(images): ##Main function to run inference on raw images
    dimension_list = []
    processedimages = []
    for i in range(images.shape[0]):
        image = images[i]
        size = image[:,:,0].shape
        [x,y,w,h] ,cropdim,limg = preprocess(image,(256,256))
        [croprow,cropcol,col] = cropdim
        if col==0:
            print(i, 'single')
        else:
            print(i, 'double')
        dimension_list.append([size,x,y,w,h,croprow,cropcol,col,limg])
        for k in range(len(limg)):
            processedimages.append(limg[k])
        print('pre-processing ' + str(i) + ' completed')
    processedimages = np.asarray(processedimages)
    print(processedimages.shape)
    mask = []
    processedimages = processedimages/255
    if not processedimages.shape[0]>30:
        masks = (model_a.predict(processedimages))
        masks = masks>0.3
    else:
        masks = np.zeros((0,256,256,1),dtype = bool)
        tot = processedimages.shape[0]
        k = int(tot/20)
        for a in range(k):
            if not a==k-1:
                mask = (model_a.predict(processedimages[a*20:(a+1)*20,:,:,:]))
            else:
                mask = (model_a.predict(processedimages[a*20:,:,:,:]))
            mask = mask>0.3
            masks = np.vstack((masks,mask))
    print('masks generated')
    masks = masks[:,:,:,0]
    j=0
    final_masks = []
    for i in range(len(dimension_list)):
        [size,x,y,w,h,croprow,cropcol,col,limg] = dimension_list[i]
        if len(limg)==1:
            msk = np.zeros((size))
            mask = cv2.resize(masks[j].astype(np.uint8), (cropcol,croprow), interpolation = cv2.INTER_NEAREST)
            msk[y:y+h,x:x+w] = mask
            final_masks.append(msk)
        else:
            msk = np.zeros((size),dtype = bool)
            mask = np.column_stack((masks[j],masks[j+1]))
            mask = cv2.resize(mask.astype(np.uint8), (cropcol,croprow), interpolation = cv2.INTER_NEAREST)
            j = j+1
            msk[y:y+h,x:x+w] = mask
            final_masks.append(msk)
        j = j+1
    final_masks = np.asarray(final_masks,dtype = object)
    return final_masks



