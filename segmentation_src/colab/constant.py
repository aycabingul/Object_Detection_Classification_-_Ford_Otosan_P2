#libraries imported
import os

IMG_DIR='data/images'

#Write the file path of the mask file to mask_dir
MASK_POLYGON_DIR  ='data/mask_polygon'
#If there is no file in the given file path, a new file is created
if not os.path.exists(MASK_POLYGON_DIR): 
    os.mkdir(MASK_POLYGON_DIR)

#Write the file path of the mask file to mask_dir
MASK_LINE_DIR  ='data/mask_line'
#If there is no file in the given file path, a new file is created
if not os.path.exists(MASK_LINE_DIR): 
    os.mkdir(MASK_LINE_DIR)

JSON_DIR = 'data/jsons'#The path to the file is assigned to the variable


    
predict='data/predict'
if not os.path.exists(predict):
    os.mkdir(predict)
    
    
    
best_models='models'
if not os.path.exists(best_models):
    os.mkdir(best_models)
