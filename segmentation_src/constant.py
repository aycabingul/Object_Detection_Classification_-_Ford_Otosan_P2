#libraries imported
import os

IMG_DIR='/mnt/sdb2/Intern_2/intern_Data/img'

#Write the file path of the mask file to mask_dir
MASK_POLYGON_DIR  ='/mnt/sdb2/Intern_2/intern_Data/mask_polygon'
#If there is no file in the given file path, a new file is created
if not os.path.exists(MASK_POLYGON_DIR): 
    os.mkdir(MASK_POLYGON_DIR)

#Write the file path of the mask file to mask_dir
MASK_LINE_DIR  ='/mnt/sdb2/Intern_2/intern_Data/line'
#If there is no file in the given file path, a new file is created
if not os.path.exists(MASK_LINE_DIR): 
    os.mkdir(MASK_LINE_DIR)

JSON_DIR = '/mnt/sdb2/Intern_2/intern_Data/ann'#The path to the file is assigned to the variable


    
result='/mnt/sdb2/Intern_2/intern_Data/result'
if not os.path.exists(result):
    os.mkdir(result)