#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 00:44:19 2021

@author: aycaburcu
"""
import os
import cv2
import tqdm
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from constant import *

jsons=os.listdir(JSON_DIR)#List created with names of json files in ann folder



for json_name in tqdm.tqdm(jsons):#access the elements in the json list
    json_path = os.path.join(JSON_DIR, json_name)#Merged json_dir with json_name and created file path
    json_file = open(json_path, 'r')#file reading process
    json_dict=json.load(json_file)#Contents of json file converted to dict data type
    mask=np.zeros((json_dict["size"]["height"],json_dict["size"]["width"]), dtype=np.uint8)
   
    
    
    mask_path = os.path.join(MASK_LINE_DIR, json_name[:-5])
    # The values of the object keys in the dicts that we obtained from each 	json file have been added to the list.
    
    for obj in json_dict["objects"]:# To access each list inside the json_objs list
                         
        if obj['classTitle']=='Solid Line':
           cv2.polylines(mask,np.array([obj['points']['exterior']],dtype=np.int32),False,color=2,thickness=16)
 
        elif obj['classTitle']=='Dashed Line':       
               cv2.polylines(mask,np.array([obj['points']['exterior']],dtype=np.int32),False,color=3,thickness=16)
    
    cv2.imwrite(mask_path,mask.astype(np.uint8))#Print filled masks in mask_path with imwrite