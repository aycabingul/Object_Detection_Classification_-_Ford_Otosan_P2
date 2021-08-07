from constant import *
import glob
import tqdm 
import cv2 
from crop_sign import cropped_image

"""
Created on Sat Aug  7 22:11:25 2021

@author: aycaburcu
"""


def visualize_sign(txt_path,save_box,cropped_path,cropped=False,save=False):
    global X_up, Y_up, X_bottom, Y_bottom
    txt_file=open(txt_path,"r")
    for line in tqdm.tqdm(txt_file):
        image_path=line.split(" ")[0]
        image=cv2.imread(image_path)
        for box in line.split(" ")[1:]:
            [X_up, Y_up, X_bottom, Y_bottom, classID] = box.split(',')
            result_box=cv2.rectangle(image, (int(X_up), int(Y_up)), (int(X_bottom),int(Y_bottom)),color=((38, 255, 255)), thickness=2)
            if save==True:
                cv2.imwrite(image_path.replace("img",save_box),result_box)
            if cropped==True:
                cropped_image(image_path, X_up,Y_up, X_bottom, Y_bottom,cropped_path)
    
        
        
        
        
    
    

txt_train_path="/mnt/sdb2/Intern_2/intern_Data/train.txt"
save_box='result_train_box'
cropped_path='cropped_train_box'
visualize_sign(txt_train_path,save_box,cropped_path,cropped=True,save=True)


txt_valid_path="/mnt/sdb2/Intern_2/intern_Data/valid.txt"
save_box='result_valid_box'
cropped_path='cropped_valid_box'
visualize_sign(txt_valid_path,save_box,cropped_path,cropped=True,save=True)