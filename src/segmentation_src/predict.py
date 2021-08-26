import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
from constant import *
import tqdm
import torch
from preprocessing import tensorize_image, tensorize_mask, image_mask_check
import cv2
from train import *
from PIL import Image




#best_model_line = torch.load('/mnt/sdb2/Intern_2/src/best_model_line.pt')




def predict(test_input_path_list,model_path,model_type):
    model= torch.load(model_path)
    model=model.eval()
    input_shape=(224,224)
    cuda=True
    global outs
    for i in tqdm.tqdm(range(len(test_input_path_list))):
    
        batch_test = test_input_path_list[i:i+1]
        img=cv2.imread(batch_test[0])
        img=cv2.resize(img,(720, 453))
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs = model(test_input)
        out=torch.argmax(outs,axis=1)
        out_cpu = out.cpu()
        outputs_list=out_cpu.detach().numpy()
        mask=np.squeeze(outputs_list,axis=0)
        mask_uint8 = mask.astype('uint8')
        
        mask= cv2.resize(mask_uint8, ((img.shape[1]), (img.shape[0]),interpolation=cv2.INTER_NEAREST)
            

        mask_ind   = mask == 1
        cpy_img  = img.copy()
        
        if model_type=='predict_line':
           img[mask==1,:]=(0, 0, 255)
           img[mask==2,:]=(38, 255, 255)
        elif model_type=='predict_polygon':
            img[mask==1,:] = (255, 0, 125)
        opac_image=(img/2+cpy_img/2).astype(np.uint8)
        predict_name=batch_test[0]
        predict_path=predict_name.replace('img', model_type)
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))


#LINE Predict#
model_path='/mnt/sdb2/Intern_2/models/Unet_1.pt'
model_type='predict_line'
predict(test_input_path_list,model_path,model_type)

# #Polygon Predict#
# model_type='predict_polygon'
# model_path='/mnt/sdb2/Intern_2/models/best_polygon_model.pt'
# predict(test_input_path_list,model_path,model_type)
