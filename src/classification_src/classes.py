import numpy as np
import pandas as pd
import os
import cv2
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import random
from matplotlib.image import imread
import tqdm
np.random.seed(40)

from matplotlib import style
style.use('fivethirtyeight')


data_dir = '../../intern_Data/classification/archive/new'
train_path = '../../intern_Data/classification/archive/new/Train'
test_path_ford = '/mnt/sdb2/Intern_2/intern_Data/cropped_valid_box'


# Resizing the images to 30x30x3
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

NUM_CATEGORIES = len(os.listdir(train_path))
NUM_CATEGORIES


#Label Overview
# classes = { }

# if __name__ == "main":
#     with open("../../intern_Data/classification/classes.json", "w") as outfile: 
#         json.dump(classes, outfile) 
    

