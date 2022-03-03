#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import scipy.ndimage as ndimage
import skimage.morphology as morphology
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import copy
import math
import os
from scipy.spatial import distance
import SimpleITK as sitk
import pydicom
from tqdm import tqdm

from thyroid_inference import get_prediction


def get_full_scan(folder_path):

    files_List  = sorted(glob(folder_path + '/*.dcm', recursive = True))
    itkimage = sitk.ReadImage(files_List[0])
    rows = int(itkimage.GetMetaData('0028|0010'))
    cols = int(itkimage.GetMetaData('0028|0011'))
    ds = pydicom.dcmread(files_List[0])
    pix_spacing = [(ds[0x0018, 0x6011])[0][0x0018, 0x602C].value , (ds[0x0018, 0x6011])[0][0x0018, 0x602E].value]
    pix_spacing = np.asarray(pix_spacing)
    pix_spacing = abs(pix_spacing).round(6)
    mn = 1000
    mx = 0
    for file in tqdm(files_List):
        itkimage = sitk.ReadImage(file)
        mn = np.min([mn, int(itkimage.GetMetaData('0020|0013'))])
        mx = np.max([mx, int(itkimage.GetMetaData('0020|0013'))])
    full_scan = np.ndarray(shape=(mx-mn+1,rows,cols,3), dtype=float, order='F')

    for file in tqdm(files_List):
        img, n = dcm_image(file)
        if not img.shape[0]>1:
            n = int(n)
            full_scan[n-mn,:,:,:] = img[0,:,:,:]

    return full_scan,pix_spacing ##[x,y]

def dcm_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    ins = float(itkimage.GetMetaData('0020|0013'))
    return numpyImage, ins

def draw_rotated_text(image, angle, xy, text, fill, *args, **kwargs):
    # get the size of our image
    image = Image.fromarray(image)
    width, height = image.size
    max_dim = max(width, height)

    # build a transparency mask large enough to hold the text
    mask_size = (max_dim * 2, max_dim * 2)
    mask = Image.new('L', mask_size, 0)

    # add text to mask
    draw = ImageDraw.Draw(mask)
    draw.text((max_dim, max_dim), text, 255, *args, **kwargs)

    if angle % 90 == 0:
        # rotate by multiple of 90 deg is easier
        rotated_mask = mask.rotate(angle)
    else:
        # rotate an an enlarged mask to minimize jaggies
        bigger_mask = mask.resize((max_dim*8, max_dim*8),
                                  resample=Image.BICUBIC)
        rotated_mask = bigger_mask.rotate(angle).resize(
            mask_size, resample=Image.LANCZOS)

    # crop the mask to match image
    mask_xy = (max_dim - xy[0], max_dim - xy[1])
    b_box = mask_xy + (mask_xy[0] + width, mask_xy[1] + height)
    mask = rotated_mask.crop(b_box)

    # paste the appropriate color, with the text transparency mask
    color_image = Image.new('RGBA', image.size, fill)
    image.paste(color_image, mask)
    image = np.array(image)
    return image

def max_dia(res0,p_x,p_y,area):
    
    mask = np.zeros((res0.shape[0],res0.shape[1]),dtype = 'uint8')
    mask = np.where((res0[:,:,0]==255)&(res0[:,:,1]==0)&(res0[:,:,2]==0),1,mask)
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
            d = (math.sqrt((((ty-cy)*p_y)**2)+(((tx-cx)*p_x)**2)))
            dist.append(d)
            res0 = cv2.arrowedLine(res0, (cx,cy), (tx,ty),(0,0,255),2)
            d = round(d,2)
            text = str(d) + ' mm'
            font = ImageFont.load_default()
            #font = ImageFont.truetype("arial.ttf", 3)
            cxx = round(cx - (res0.shape[1]/2))
            txx = round(tx - (res0.shape[1]/2))
            cyy = round((res0.shape[0]/2) -cy)
            tyy = round((res0.shape[0]/2) -ty)
            if cxx>txx:
                angle = math.degrees(math.atan((cyy-tyy)/(cxx-txx)))
                res0 = draw_rotated_text(res0, angle, [tx,ty], text, (0,255,0),font = font )
            else:
                angle = math.degrees(math.atan((tyy-cyy)/(txx-cxx)))
                res0 = draw_rotated_text(res0, angle, [cx,cy], text, (0,255,0),font = font )
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 0.25
            fontColor              = (0,255,0)
            thickness              = 1
            lineType               = 2
            text = 'Area of nodule: {}mm2'.format(str(round(area[i],2)))
            res0 = cv2.putText(res0, text,(200,(300+(20*(i+1)))),font, fontScale,fontColor)#,thickness,lineType)
        else:
            dist.append('none')
    return dist,res0

def find_nodule(input_folder,mode = 'dcm'):
    if not mode == 'dcm':
        datapath = sorted(glob(input_folder+'/*',recursive = True))
        pix_spacing = [3,3]
        data = np.empty((len(datapath)),dtype = 'object')
        for i,path in enumerate(datapath):
            data[i] = cv2.imread(path)
    else:
        data,pix_spacing = get_full_scan(input_folder)
        data = data.astype(np.uint8)
        
    masks = get_prediction(data)
    masks = masks.astype(np.uint8)
    for i in range(masks.shape[0]):
        msk = masks[i]
        msk = morphology.binary_opening(msk,np.ones((5,5)))
        msk = ndimage.morphology.binary_closing(msk,structure=np.ones((5,5)))
        msk = ndimage.binary_fill_holes(msk, structure=np.ones((5,5))).astype(np.uint8)
        msk = Image.fromarray(msk)
        msk = msk.filter(ImageFilter.ModeFilter(size=13))
        msk = np.array(msk)
        masks[i] = msk
    
    p_x = pix_spacing[0]*10
    p_y = pix_spacing[1]*10
    contmasks = []
    #contorig = []
    area = []
    for i in range(masks.shape[0]):
        img = data[i,:,:,:]
        contours, _ = cv2.findContours(masks[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areaa = []
        for cnt in contours:
            areaa.append(cv2.contourArea(cnt)*(p_x*p_y))
        area.append(areaa)
        img = cv2.drawContours(np.ascontiguousarray(img, dtype=np.uint8), contours, -1, 255, 3)
        contmasks.append(img)
        #img = data[i]
        #contours, _ = cv2.findContours(origmasks[i].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #img = cv2.drawContours(img.astype(np.uint8), contours, -1, 255, 3)
        #contorig.append(img)
    contmasks = np.asarray(contmasks,dtype = 'object')
    max_diameter = []
    for i in range(masks.shape[0]):
        d,p = max_dia(contmasks[i].astype(np.uint8),p_x,p_y,area[i])
        max_diameter.append(d)
        contmasks[i] = p
    max_d = max(max_diameter)
    min_d = min(max_diameter)
    return contmasks,max_d, min_d,area

def main_func(input_folder,output_folder):
    masks,max_diamter, min_diameter,area = find_nodule(input_folder,'dcm')
    output_list = []
    classUID = []
    annotations = []
    dcmData = pydicom.dcmread("sample.dcm")
    dcmData2 = copy.copy(dcmData)
    newseriesnum = 1
    dcmData2.SeriesNumber = newseriesnum
    siu = copy.copy(dcmData.SeriesInstanceUID)
    sopiu = copy.copy(dcmData.SOPInstanceUID)
    mssiu = copy.copy(dcmData.file_meta.MediaStorageSOPInstanceUID)
    dcmData2.SeriesInstanceUID= siu +  '.' + str(newseriesnum)
    for i in range(masks.shape[0]):
        mask = masks[i]
        new_img = mask.astype(np.uint8)
        dcmData2.PixelData = new_img.tobytes()
        dcmData2.Rows = new_img.shape[0]
        dcmData2.Columns = new_img.shape[1]
        dcmData2.InstanceNumber = i+1
        dcmData2.SOPInstanceUID = sopiu + '.' + str(newseriesnum) +  str(i+1).zfill(2)
        try:
            dcmData2.file_meta.MediaStorageSOPInstanceUID = mssiu + '.' + str(newseriesnum) +  str(i+1).zfill(3) 
        except:
            error=0
        name = 'UltrasoundSlice_'  + str(i+1)  + '.dcm'
        name = os.path.join(output_folder,name)
        output_list.append(name)
        classUID.append(str(dcmData2.SOPClassUID))
        annotations.append('None')
        dcmData2.save_as(name)
    return output_list, classUID,annotations , max_diamter , min_diameter,area




def predictions(input_folder,output_folder):
    output_list, classUID,annotations, max_diameter, min_diameter,area = main_func(input_folder,output_folder)
    mimeType = "application/dicom"
    recommendation_string = {"finding": "Maximum diameter of the nodule is: " + str(max_diameter)+ "mm","conclusion":"conclusion","recommendation":"recommendation"} 
    #device=cuda.get_current_device()
    #device.reset()
    return output_list, classUID, mimeType, recommendation_string,annotations

