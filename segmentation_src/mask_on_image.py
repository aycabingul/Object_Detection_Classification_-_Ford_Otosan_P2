import cv2 
import os
import numpy as np
import tqdm
from constant import *


masks_name_polygon=os.listdir(MASK_POLYGON_DIR)
masks_name_line=os.listdir(MASK_LINE_DIR)

#Convert images in jpg format to png format
#for maskname in tqdm.tqdm(masks_name):
     #img=cv2.imread(os.path.join(IMG_DIR,maskname[:-4]+".jpg")).astype(np.uint8)
     #cv2.imwrite(os.path.join(IMG_DIR,maskname),img)
     #os.remove(os.path.join(IMG_DIR,maskname[:-4]+".jpg"))#Delete jpg after saving as png
     
     
for maskname in tqdm.tqdm(masks_name_polygon):#Access individual elements of the masks_name list
    img_name=maskname[:-3]+"jpg"
    img=cv2.imread(os.path.join(IMG_DIR,img_name)).astype(np.uint8)
    mask_polygon=cv2.imread(os.path.join(MASK_POLYGON_DIR,maskname),0).astype(np.uint8)
    mask_line=cv2.imread(os.path.join(MASK_LINE_DIR,maskname),0).astype(np.uint8)
    
    mask_ind   = mask_polygon == 1
    

    cpy_img  = img.copy()
    
    img[mask_polygon==1,:] = (255, 0, 125)
    img[mask_line==2,:]=(0, 0, 255)
    img[mask_line==3,:]=(38, 255, 255)
    img[mask_polygon==4,:]=(0,140,255)
    img[mask_polygon==5,:]=(0,0,255)
    
    
    
    opac_image=(img/2+cpy_img/2).astype(np.uint8)
    
    cv2.imwrite(os.path.join(result,maskname),opac_image)#save
 

