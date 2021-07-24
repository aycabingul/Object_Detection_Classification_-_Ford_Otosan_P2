
import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
from constant import *
import tqdm


best_model_line = torch.load('/mnt/sdb2/Intern_2/src/best_model_line.pt')
best_model_polygon= torch.load('/mnt/sdb2/Intern_2/src/best_model_polygon.pt')

def predict(test_input_path_list):

    for i in tqdm.tqdm(range(len(test_input_path_list))):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs_line = best_model_line(test_input)
        outs_polygonbest_model_polygon(test_input)
        out_line=torch.argmax(outs_line,axis=1)
        out_polygon=torch.argmax(outs_polygon,axis=1)
        out_line_cpu = out_line.cpu()
        out_ploygon_cpu = out_ploygon.cpu()
        outputs_line_list=out_line_cpu.detach().numpy()
        outputs_ploygon_list=out_polygon_cpu.detach().numpy()
        
        mask=np.squeeze(outputs_list,axis=0)
            
            
        img=cv2.imread(batch_test[0])
        mg=cv2.resize(img,(224,224))
        mask_ind   = mask == 1
        cpy_img  = mg.copy()
        mg[mask==1 ,:] = (255, 0, 125)
        opac_image=(mg/2+cpy_img/2).astype(np.uint8)
        predict_name=batch_test[0]
        predict_path=predict_name.replace('image', 'predict')
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))

predict(test_input_path_list)


    

# zip:
#letters = ['a', 'b', 'c']
#numbers = [0, 1, 2]
#for l, n in zip(letters, numbers):
    #print(f'Letter: {l}')
    #print(f'Number: {n}')
# Letter: a
# Number: 0
# Letter: b
# Number: 1
# Letter: c
# Number: 2