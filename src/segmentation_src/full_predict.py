import torch.nn as nn
import torch.optim as optim
from constant import *
import tqdm
import torch
from preprocessing import tensorize_image, tensorize_mask, image_mask_check
import cv2
from train import *
from PIL import Image



model_freespace= torch.load('/mnt/sdb2/Intern_2/models/Unet_1.pt')
model_line=torch.load('/mnt/sdb2/Intern_2/models/best_line_model.pt',map_location='cuda:0')
model_freespace=model_freespace.eval()
model_line=model_line.eval()
input_shape=(224,224)
cuda=True
if cuda:
    model_line = model_line.cuda()
    model_freespace=model_freespace.cuda()


for i in tqdm.tqdm(range(len(test_input_path_list))):
    batch_test = test_input_path_list[i:i+1]
    img=cv2.imread(batch_test[0])
    
    test_input_line = tensorize_image(batch_test, input_shape, cuda)
    test_input_freespace=tensorize_image(batch_test, input_shape, cuda)
    
    outs_freespace = model_freespace(test_input_freespace)
    outs_line = model_line(test_input_line)
    
    out_freespace=torch.argmax(outs_freespace,axis=1)
    out_line=torch.argmax(outs_line,axis=1)
    
    out_freespace_cpu = out_freespace.cpu()
    out_line_cpu=out_line.cpu()
    
    outputs_list_freespace=out_freespace_cpu.detach().numpy()
    outputs_list_line=out_line_cpu.detach().numpy()
    
    mask_freespace=np.squeeze(outputs_list_freespace,axis=0)
    mask_line=np.squeeze(outputs_list_line,axis=0)
    
    mask_uint8_line = mask_line.astype('uint8')
    mask_uint8_freespace = mask_freespace.astype('uint8')
    
    mask_line= cv2.resize(mask_uint8_line, ((img.shape[1]), (img.shape[0])),interpolation=cv2.INTER_NEAREST)
    mask_freespace= cv2.resize(mask_uint8_freespace, ((img.shape[1]), (img.shape[0])),interpolation=cv2.cv2.INTER_CUBIC)
        


    mask_ind   = mask_line == 1
    mask_ind   = mask_freespace == 1
    cpy_img  = img.copy()
    
    img[mask_freespace==1,:] = (255, 0, 125)


    img[mask_line==1,:]=(0, 0, 255)
    img[mask_line==2,:]=(38, 255, 255)
    
    
    opac_image=(img/2+cpy_img/2).astype(np.uint8)
    predict_name=batch_test[0]
    predict_path=predict_name.replace('images', 'predict')
    cv2.imwrite(predict_path,opac_image.astype(np.uint8))

