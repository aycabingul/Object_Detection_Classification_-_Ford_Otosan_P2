from classes import *
import numpy as np
import cv2
import json
import os
import torch
import tqdm 
from matplotlib import pyplot as plt
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage import transform
import random 
from torchvision import transforms as T
from PIL import Image
num_list=[32,31,30,28,27,26,25,12,6]
for i in tqdm.tqdm(num_list):
    path = data_dir + '/Train/' + str(i)
    images = os.listdir(path)

    for image in images:
        img=Image.open(path+'/'+image)
        color_aug = T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06)
        img_aug = color_aug(img)
        new_path=path+'/'+image[:-4]+"-1"+".png"
        img_aug=np.array(img_aug)
        cv2.imwrite(new_path,img_aug)
        
        flipUD = np.flipud(img)
        new2_path=path+'/'+image[:-4]+"-2"+".png"
        cv2.imwrite(new2_path,flipUD )
        
        rotated = rotate(np.array(img),45)
    
        new3_path=path+'/'+image[:-4]+"-3"+".png"
        cv2.imwrite(new3_path,rotated)

    
        transform = AffineTransform(translation=(25,25))
        wrapShift = warp(np.array(img),transform,mode='wrap')
        new4_path=path+'/'+image[:-4]+"-4"+".png"
        cv2.imwrite(new4_path,wrapShift)

        sigma=0.155

        noisyRandom = random_noise(np.array(img),var=sigma**2)
        new5_path=path+'/'+image[:-4]+"-5"+".png"
        cv2.imwrite(new5_path,noisyRandom)
    
