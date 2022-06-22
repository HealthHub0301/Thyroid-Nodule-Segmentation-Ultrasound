#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"


# In[2]:


import pydicom
import numpy as np
from scipy import ndimage
from scipy.spatial import distance
import rasterio
from rasterio import features
import shapely
from shapely.geometry import Point, Polygon
import cv2
import matplotlib.pyplot as plt
import sys
# from numpyencoder import NumpyEncoder
import json
from glob import glob
import tensorflow as tf


# In[45]:


import tensorflow.keras.backend as K
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


# In[46]:


def get_indi_F_patch(x_min, x_max, max_limit, min_limit=0, length=70):
    if((max_limit-min_limit)<length):
        print('Your diminsions are less than required length')
        return(None,None)
    if((x_max-x_min)>length):
        x_max = x_min+length
    if(x_min==x_max):
        x_max = x_max+1
#     print(length-(x_max-x_min))
    fact = (length-(x_max-x_min))/2
#     print(fact)
    max_ind = x_max+fact
    
    if(max_ind>max_limit):
        return(max_limit-length,max_limit)
    
    min_ind = x_min-fact
    if(min_ind<min_limit):
        return(0,length)
    return(int(min_ind),int(max_ind))


# In[47]:


def get_mask2(mask):
    R1 = random.randint(-2, 10)
    R2 = random.randint(-2, 10)
    b,c = np.where(mask>0)
    xmax = np.min([np.max(b)+R1,420])
    xmin = np.max([np.min(b)+R1,0])
    ymax = np.min([np.max(c)+R2,400])
    ymin = np.max([np.min(c)+R2,0])
    mask = mask.copy()
    mask = mask*0
    mask[xmin:xmax,ymin:ymax]=1
    return(mask)


# In[48]:


def _generate_Xy(ID,aug):
    """Generates data containing batch_size images
    :param list_IDs_temp: list of label ids to load
    :return: batch of images
    """
    # Initialization
    X1 = np.empty((1,256,256,1))
    X2 = np.empty((1,256,256,1))
    y = np.empty((1,256,256,1))
    # Generate data
    img1 = images[ID].copy()
    mask = masks[ID].copy()
    img1=img1/np.max(img1)
    mask=mask/np.max(mask)
    img2 =  get_mask2(mask[:,:,0])
#             inps_final.append([img1,img2,img3,img4])
#     aug = random.randint(0, 3)
    if(aug==3):
        img1 = np.flip(np.rot90(img1))
        img2 = np.flip(np.rot90(img2))
        mask = np.flip(np.rot90(mask))
    if(aug==1):
        img1 = np.rot90(img1)
        img2 = np.rot90(img2)
        mask = np.rot90(mask)
    if(aug==2):
        img1 = np.flip(img1)
        img2 = np.flip(img2)
        mask = np.flip(mask)
    pos_x = np.where(img2>0)[0]
    xmin,xmax = np.min(pos_x),np.max(pos_x)
    pos_y = np.where(img2>0)[1]
    ymin,ymax = np.min(pos_y),np.max(pos_y)
    xmini,xmaxi = get_indi_F_patch(xmin,xmax, max_limit=420, min_limit=0, length=256)
    ymini,ymaxi = get_indi_F_patch(ymin,ymax, max_limit=400, min_limit=0, length=256)
    X1[0,:,:,0]=img1[xmini:xmaxi,ymini:ymaxi,0]
    X2[0,:,:,0]=img2[xmini:xmaxi,ymini:ymaxi]
    y[0,:,:,0] = mask[xmini:xmaxi,ymini:ymaxi,0]
    return X1,X2,y


# In[49]:


def dice_coefficient(predicted, target):
    smooth = 1
    product = np.multiply(predicted, target)
    intersection = np.sum(product)
    coefficient = (2 * intersection + smooth) / (np.sum(predicted) + np.sum(target) + smooth)
    return coefficient


# In[50]:


def get_scan_ROI(img,img2):
    pos_x = np.where(img2>0)[0]
    xmin,xmax = np.min(pos_x),np.max(pos_x)
    pos_y = np.where(img2>0)[1]
    ymin,ymax = np.min(pos_y),np.max(pos_y)
    xmini,xmaxi = get_indi_F_patch(xmin,xmax, max_limit=int(img.shape[0]), min_limit=0, length=256)
    ymini,ymaxi = get_indi_F_patch(ymin,ymax, max_limit=int(img.shape[1]), min_limit=0, length=256)
    X1 = np.empty((1,256,256,1))
    X2 = np.empty((1,256,256,1))
    X1[0,:,:,0]=img[xmini:xmaxi,ymini:ymaxi,0]
    X2[0,:,:,0]=img2[xmini:xmaxi,ymini:ymaxi,0]
    return X1,X2,[xmini,xmaxi,ymini,ymaxi]


# In[51]:


def max_dia(mask):    
#     mask = np.zeros((res0.shape[0],res0.shape[1]),dtype = 'uint8')
#     mask = np.where((res0[:,:,0]==255)&(res0[:,:,1]==0)&(res0[:,:,2]==0),1,mask)
    mask, num_lab = ndimage.label(mask)
    dist = []
    for i in range(num_lab):
        points = np.argwhere(mask==i+1)
        points = points[points[:,0].argsort()]
        distances = distance.cdist(points,points)
        if len(distances)>1:
            [p1,p2] = np.squeeze(np.where(distances == np.max(distances)))
            if isinstance(p1, np.ndarray): 
                p1 = p1[-1]
            if isinstance(p2, np.ndarray): 
                p2 = p2[-1]
            [ty,tx] = points[p1]
            [cy,cx] = points[p2]
            dist.append([tx,ty,cx,cy])
        else:
            dist.append('none')
    return dist


# In[52]:


def mask_to_polygons_layer(mask):
    most_polygons = []
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask >0), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        shh = shape['coordinates']
        for k in range(len(shh)):
            sh = shh[k]
            sh = np.squeeze(sh)
            sh = np.asarray(sh)
            sh = sh.flatten()
            #sh = sh.astype(np.uint16)
            sh = list(sh)
            most_polygons.append(sh)
    return most_polygons


# In[53]:


def get_results(path,Lists):
    dcm = pydicom.dcmread(path)
    temp = dcm.pixel_array
    temp = temp-np.min(temp)/(np.max(temp)-np.min(temp))
    img2 = np.zeros(temp.shape)
    result = np.zeros((temp.shape[0],temp.shape[1]))
    img2[Lists[0]:Lists[1],Lists[2]:Lists[3]]=1
    print(temp.shape)
    print(Lists)
    img1,img2,indis = get_scan_ROI(temp,img2)
    model = tf.keras.models.load_model('thyroid_1.hdf5', custom_objects={'dice_coef': dice_coef})
    z = model.predict([img1,img2])
    result[indis[0]:indis[1],indis[2]:indis[3]]=z[0,:,:,0]
    area = np.sum(result)
    max_diameter = max_dia(result)
    return(result,max_diameter,area,dcm)


# In[54]:


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)


# In[55]:


def get_cord(path,additional_parameters):
    a = glob(path+'/*.dcm')
    UID_paths = {}
    for i in range(len(a)):
        dcm = pydicom.dcmread(a[i])
        try:
            UID = dcm.SOPInstanceUID
        except:
            UID = dcm.file_meta.MediaStorageSOPInstanceUID
        UID_paths[UID] = a[i]
    lists = additional_parameters['list']
    final_list = []
    for i in range(len(lists)):
        ROI = lists[i]['position']
        ROIs = [ROI[0],ROI[2],ROI[1],ROI[3]]
        Cur_UID = lists[i]['sopInstanceUID']
        Cur_path = UID_paths[Cur_UID]
        final_list.append([Cur_path,ROIs])
    return(final_list)


# In[56]:


def main_call(inputpath,outpath,additional_parameters):
    annotations = []
    output_List= []
    class_UID = []
    mimeType = []
    annotations=[]
    recomendation_string = {
        "finding":"dummy findings",
        "conclusion":"dummy conclusion",
        "recommendation":"dummy recommendation"
    }
    
    Lists = get_cord(inputpath,additional_parameters)
    for List in Lists:
        result,max_diameter,area,dcm = get_results(List[0],List[1])
        print(max_diameter)
        diameter = max_diameter[1]
        poly = mask_to_polygons_layer(result)
        ann = []        
        colour = '#ac2d2d'
        dictionary = {'properties':{'color':colour,
                                   'position': poly},
                      'isFlipH' : False,
                      'angle' : 0,
                      'type' : 'polygon',
                      'isFlipV' : False}
        ann.append(dictionary)
        dictionary = {'properties':{'color':colour,
                                   'position': [float(x) for x in diameter]},
                      'item':[{
                          'properties':{'color': colour,
                                        'position': [diameter[0] , diameter[1]],
                                        'fontSize':'15',
                                        'font': 'arial'},
                                      'type' : 'text'}],
                      'isFlipH' : False,
                      'angle' : 0,
                      'type' : 't_ruler',
                      'isFlipV' : False}
        ann.append(dictionary)
        pos = [20.0,20.0]
        dictionary = {'properties':{'color':colour,
                                   'position':pos,
                                   'font': 'arial',
                                   'fontSize': '20'},
                      'text': 'nodule Area: ' + str(area) + ' mm' + get_super('2'),
                      'isFlipH' : False,
                      'angle' : 0,
                      'type' : 'text',
                      'isFlipV' : False}
        annotations.append(ann)
        try:
            class_UID.append(dcm.SOPInstanceUID)
        except:
            class_UID.append(dcm.file_meta.MediaStorageSOPInstanceUID)

    mimeType.append(None)
    output_List.append(None)

       
    all_result={"output_list":output_List, "classUID":class_UID, "mimeType":mimeType, "recommendation_string": recomendation_string,"annotations":annotations}
    #path_new=path
    print("values.sjon path",inputpath)
    with open(inputpath+"values.json", "w") as outfile:
    #with open("/home/azka/HPACS_v3/Lung_Nodule_ROI/input_files/lnq1/"+"values.json", "w") as outfile:
        json.dump(all_result, outfile ,cls=NumpyEncoder)  
    return(output_List,class_UID,mimeType,recomendation_string,annotations)


# In[57]:


def main(input_path,outpath,):
    with open(input_path+'additional_paras.json', 'r') as openfile:
        additional_parameters = json.load(openfile)
    main_call(input_path,outpath,additional_parameters)


# In[31]:


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])


# In[61]:


# additional_parameters = {'list':[{ "sopInstanceUID":'1.2.392.200036.9116.6.18.14612120.3972.20160528051038563.1.423'  ,
# "position": [350,300,450,410]
#               },
#         { "sopInstanceUID":'1.2.392.200036.9116.6.18.14612120.3972.20160528051038563.1.423',
# "position": [350,300,450,410]
#               }]}
# outpath = '/home/None/'
# input_path = '/home/abdullah/Thyroid_nodule/data/thyroid_new_ultrasound_scans/Ultra Sound Scans/HH Raw Scans/rem/Test Thyroid scans from HH without Dr Kim/'
# output_List,class_UID,mimeType,recomendation_string,annotations = main_call(input_path,outpath,additional_parameters)


# In[ ]:




