import numpy as np
import cv2
import json
import os
import torch
import tqdm 
import glob

from constant import *


batch_images=[] 
image_path=[] 
for name in os.listdir(IMG_DIR):
    image_path.append(os.path.join(IMG_DIR,name))
#Save the names of the files in the folder in IMAGE_DIR path to the list

mask_polygon_path=[] 
for name in os.listdir(MASK_POLYGON_DIR):
    mask_polygon_path.append(os.path.join(MASK_POLYGON_DIR,name))
#Save the names of the files in the folder in MASK_DIR path to the list

mask_line_path=[] 
for name in os.listdir(MASK_LINE_DIR):
    mask_line_path.append(os.path.join(MASK_LINE_DIR,name))
#Save the names of the files in the folder in MASK_DIR path to the list
output_shape=[224,224]
batch_masks=[]


def tensorize_image(image_path,output_shape,cuda=False): #Create a 2-parameter function
    batch_images=[] 
    for image in image_path[:8]: #Access the elements in the image_path list one by one with the for loop
        img=cv2.imread(image) #Read file in path assigned to image variable
        norm_img = np.zeros((1920,1208))
        final_img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
        img=cv2.resize(final_img,tuple(output_shape),interpolation = cv2.INTER_NEAREST) #Apply image resize operation
        torchlike_image = torchlike_data(img)
        batch_images.append(torchlike_image) #save resized images to list
    torch_image=torch.as_tensor(batch_images,dtype=torch.float32).float()#convert the list to torch tensor
    
    
    if cuda:
        torch_image= torch_image.cuda()
    return torch_image
    print(torch_image.size())
    #[4765,300,300,3] 




def tensorize_mask(mask_path,output_shape,n_classes,cuda=False):#Create a 2-parameter function
    batch_masks=[]
    for mask in mask_path:#Access the elements in the image_path list one by one with the for loop
        mask=cv2.imread(mask,0)
        #a change here; Read as (HXW) (black, white)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)    
        mask= cv2.resize(mask, tuple(output_shape),interpolation = cv2.INTER_NEAREST)#Apply image resize operation
        #one hot encode 
        mask=one_hot_encoder(mask,n_classes)
        torchlike_mask = torchlike_data(mask)
        batch_masks.append(torchlike_mask)
    torch_mask=torch.as_tensor(batch_masks,dtype=torch.float32)

    
    if cuda:
        torch_mask=torch_mask.cuda()
    return torch_mask
    #[4765,300,300,2] 


def one_hot_encoder(res_mask,n_classes):
    #one hot encode
    #Create an np.array of zeros.
    one_hot=np.zeros((res_mask.shape[0],res_mask.shape[1],n_classes),dtype=np.int)
    #Find unique values in res_mask [0,1]
    #increase in i by the length of the list
    #[0,1] when returning the inside of list, each list element is given to unique_value variable
    for i,unique_value in enumerate(np.unique(res_mask)):
        one_hot[:,:,i][res_mask==unique_value]=1
    return one_hot

def torchlike_data(data):
    #transpose process 
    n_channels = data.shape[2]
    torchlike_data = np.empty((n_channels, data.shape[0], data.shape[1]))#Returns a new array of the given shape and type.
    #creates an array of these sizes
    for ch in range(n_channels):# generates ch numbers as long as the list
        torchlike_data[ch] = data[:,:,ch] #torchlike_data[0]=data[:,:,0] 
        #Export data in data individually to torchlike_data
    return torchlike_data

#If the mask name and image name match, it looks at it, if it does not, it displays a warning.
def image_mask_check(image_path_list, mask_path_list):
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        mask_name  = mask_path.split('/')[-1].split('.')[0]
        assert image_name == mask_name, "Image and mask name does not match {} - {}".format(image_name, mask_name)


if __name__ == '__main__':
    output_shape=[224,224]
    
    #The path to the mask folder is assigned to the variable
    image_path=glob.glob(os.path.join(IMG_DIR, '*'))
    image_path.sort()
    mask_polygon_path=glob.glob(os.path.join(MASK_POLYGON_DIR, '*'))
    mask_polygon_path.sort()
    mask_line_path=glob.glob(os.path.join(MASK_LINE_DIR, '*'))
    mask_line_path.sort()
    image_mask_check(image_path,mask_polygon_path)
    image_mask_check(image_path,mask_line_path)
    
    #Save the names of the files in the folder in IMAGE_DIR path to the list
    batch_image_list=image_path[:4]
    batch_image_tensor=tensorize_image(batch_image_list,output_shape)#call function
    
    batch_polygon_list=mask_polygon_path[:4]
    batch_polygon_tensor=tensorize_mask(batch_polygon_list,output_shape,4)#call function
    
    
    batch_line_list=mask_line_path[:4]
    batch_line_tensor=tensorize_mask(batch_line_list,output_shape,3)#call function
    
    
    #The names of the files in the folder in the MASK_DIR path have been combined into a list via mask_dir.
    print(batch_image_list)
    print(batch_image_tensor.dtype)
    print(type(batch_image_tensor))
    print(batch_image_tensor.shape)
    
    print(batch_polygon_list)
    print(batch_polygon_tensor.dtype)
    print(type(batch_polygon_tensor))
    print(batch_polygon_tensor.shape)  
    
    print(batch_line_list)
    print(batch_line_tensor.dtype)
    print(type(batch_line_tensor))
    print(batch_line_tensor.shape)  