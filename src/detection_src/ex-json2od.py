import os
import cv2
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from constant import *



def json2od(JSON_DIR,json_name):

    
    json_path = os.path.join(JSON_DIR, json_name)#Merged json_dir with json_name and created file path
    json_file = open(json_path, 'r')#file reading process
    json_dict=json.load(json_file)#Contents of json file converted to dict data type
    
       
    if len(json_dict["objects"]) == 0:
            a='obje bulunamadı'
            return a
    obj_id=0
    annotations = []
    for obj in json_dict["objects"]:# To access each list inside the json_objs list
        
        if obj['classTitle']=='Traffic Sign':
     
               
                # Eliminate small ones 
                if (obj['points']['exterior'][1][0] - obj['points']['exterior'][0][0]) < 16 or (obj['points']['exterior'][1][1] - obj['points']['exterior'][0][1]) < 16:
                    continue
                
                # Add into list
                annotations.append(str(obj['points']['exterior'][0][0]) + ',' + str(obj['points']['exterior'][0][1]) + ',' + str(obj['points']['exterior'][1][0]) + ',' + str(obj['points']['exterior'][1][1]) + ',' + str(obj_id))
                
     # Modify list
    strlabel = ''
    for idx in range(len(annotations)):
        if idx != 0:
            strlabel += ' '

        strlabel += annotations[idx]

    return strlabel




""" Write down into the txt file """
train_txt=open(OD_TRA_LABEL, "w+")
test_txt=open(OD_TES_LABEL, "w+")
valid_txt=open(OD_VAL_LABEL, "w+")
jsons=os.listdir(JSON_DIR)#List created with names of json files in ann folde
param=0
valid_size = 0.3#Validation dataset is used to evaluate a particular model, but this is for frequent evaluation.
test_size  = 0.05#rate of data to be tested
test_ind  = int(3671 * test_size)#Multiply indices length by test_size and assign it to an int-shaped variable
valid_ind = int(test_ind + 3671 * valid_size)
line_list=[]
for json_name in tqdm.tqdm(jsons):
    
    image_name = os.path.splitext(json_name)[0]
    # Change from png to jpg
    image_name=image_name[:-3]+'jpg'
    image_path = os.path.join(IMG_DIR, image_name)

    if len(json2od(JSON_DIR,json_name))!=0:
        line = image_path+' '+json2od(JSON_DIR,json_name)+'\n'
        line_list.append(line)
        
        
        if param<test_ind:
        # Write down
            test_txt.write(line)
        elif param>=test_ind and param<valid_ind:
            valid_txt.write(line)
        elif param>=valid_ind:
            train_txt.write(line)
        
        param=param+1

        
# Close txt file
train_txt.close()
test_txt.close()
valid_txt.close()



