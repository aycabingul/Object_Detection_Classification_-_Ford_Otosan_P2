#libraries imported
import os

IMG_DIR='/mnt/sdb2/Intern_2/intern_Data/img'

JSON_DIR = '/mnt/sdb2/Intern_2/intern_Data/ann'#The path to the file is assigned to the variable

OD_TRA_LABEL='/mnt/sdb2/Intern_2/intern_Data/train.txt'  
OD_TES_LABEL='/mnt/sdb2/Intern_2/intern_Data/test.txt'  
OD_VAL_LABEL='/mnt/sdb2/Intern_2/intern_Data/valid.txt'  

result_train_box ='/mnt/sdb2/Intern_2/intern_Data/result_train_box'
result_valid_box ='/mnt/sdb2/Intern_2/intern_Data/result_valid_box'
#If there is no file in the given file path, a new file is created
if not os.path.exists(result_valid_box): 
    os.mkdir(result_valid_box)

if not os.path.exists(result_train_box): 
    os.mkdir(result_train_box)
    
cropped_path_valid='/mnt/sdb2/Intern_2/intern_Data/cropped_valid_box'
cropped_path_train ='/mnt/sdb2/Intern_2/intern_Data/cropped_train_box'
#If there is no file in the given file path, a new file is created
if not os.path.exists(cropped_path_valid): 
    os.mkdir(cropped_path_valid)

if not os.path.exists(cropped_path_train): 
    os.mkdir(cropped_path_train)