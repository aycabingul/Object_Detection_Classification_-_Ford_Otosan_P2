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
num_list=[26]
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
        

    

