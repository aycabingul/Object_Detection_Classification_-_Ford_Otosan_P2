from line_Unet import LINE_NET
from polygon_model import POL_NET
from line_SegNet import SegNet
from preprocessing import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
from constant import *
import tqdm
import torch
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

def train(valid_size,test_size,batch_size,epochs,cuda,input_shape,n_classes,mask_dir,model,model_save,train_if):
    global test_input_path_list
    global train_input_path_list
    ######### DIRECTORIES #########
    SRC_DIR = os.getcwd()
    
    ###############################
    
    
    # PREPARE IMAGE AND MASK LISTS
    image_path_list = glob.glob(os.path.join(IMG_DIR, '*'))
    image_path_list.sort()
    
    mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
    mask_path_list.sort()
    
    # DATA CHECK
    image_mask_check(image_path_list, mask_path_list)
    #Checked whether the elements in mask_path_list and image_path_list list are the same.
    
    
    
    # SHUFFLE INDICES
    indices = np.random.permutation(len(image_path_list))
    #A random array of permutations for the length of the image_path_list steps_per_epoch = len (train_input_path_list) // batch_size is created
    
    
    # DEFINE TEST AND VALID INDICES
    test_ind  = int(len(indices) * test_size)#Multiply indices length by test_size and assign it to an int-shaped variable
    valid_ind = int(test_ind + len(indices) * valid_size)
    
    # SLICE TEST DATASET FROM THE WHOLE DATASET
    test_input_path_list = image_path_list[:test_ind] #Get 0 to 476 elements of the image_path_list list
    test_label_path_list = mask_path_list[:test_ind]#Get 0 to 476 elements of the mask_path_list list
    
    # SLICE VALID DATASET FROM THE WHOLE DATASET
    valid_input_path_list = image_path_list[test_ind:valid_ind]#Get 476 to 1905 elements of the image_path_list list
    valid_label_path_list = mask_path_list[test_ind:valid_ind]#Get 476 to 1905 elements of the mask_path_list list
    
    # SLICE TRAIN DATASET FROM THE WHOLE DATASET
    train_input_path_list = image_path_list[valid_ind:]#Get the elements of the image_path_list list from 1905 to the last element
    train_label_path_list = mask_path_list[valid_ind:]#Get the elements of the mask_path_list list from 1905 to the last element
    #burada yukarıda vermiş olduğumuz test verisi için tüm datanın 0.1 ve validation verisi tüm datanın 0.3 içermeli
    #Here, for the test data we have given above, all the data should contain 0.1 and all the validation data should contain 0.3, 
    #but both of them do not belong to the same data data.
    
    # train_input_path_list.extend(aug_path_list)
    # train_label_path_list.extend(aug_mask_path_list)
    
    
    
    
    
    if train_if:    
        steps_per_epoch = len(train_input_path_list)//batch_size
        # Find how many times to do it by dividing the length of the train data (training data) by batch_size
        #in an epoch, a data string in the dataset goes to the end in neural networks
        #It then waits there until the batch reaches you, the error rate is calculated after the data reaches the end
        #Divide the training data set by 4 since our batch_size is 4
        
        # CALL MODEL
        model = model
        #Enter parameters into model and assign output to variable
        
        # DEFINE LOSS FUNCTION AND OPTIMIZER
        
    
        criterion = nn.CrossEntropyLoss()#Creates a criterion that measures the Binary Cross Entropy between target and output:
        #BCELoss is an acronym for Binary CrossEntropyLoss, a special case of BCOMoss CrossEntropyLoss used only for two categories of problems.
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        #Commonly used momentum beta coefficient is 0.9.
        #lr=learning rate
        
        # IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
        if cuda:
            model = model.cuda()
        
        val_losses=[]
        train_losses=[]
        # TRAINING THE NEURAL NETWORK
        for epoch in tqdm.tqdm(range(epochs)):
        
            running_loss = 0
            #In each epoch, images and masks are mixed randomly in order not to output images sequentially.
            pair_IM=list(zip(train_input_path_list,train_label_path_list))
            np.random.shuffle(pair_IM)
            unzipped_object=zip(*pair_IM)
            zipped_list=list(unzipped_object)
            train_input_path_list=list(zipped_list[0])
            train_label_path_list=list(zipped_list[1])
            
            for ind in tqdm.tqdm(range(steps_per_epoch)):
                batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
                #train_input_path_list [0: 4] gets first 4 elements on first entry
                #in the second loop train_input_list [4: 8] gets the second 4 elements
                #element advances each time until batch_size
                batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
                batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
                batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
                #Our data that we will insert into the model in the preprocess section is prepared by entering the parameters.
                
                optimizer.zero_grad()#gresets the radian otherwise accumulation occurs on each iteration
                # Manually reset gradients after updating Weights
                
                
                outputs = model(batch_input) # Give the model batch_input as a parameter and assign the resulting output to the variable.
                
    
                # Forward passes the input data
                batch_label= torch.argmax(batch_label, dim=1)
                loss = criterion(outputs, batch_label)
                loss.backward()# Calculates the gradient, how much each parameter needs to be updated
                optimizer.step()# Updates each parameter according to the gradient
        
                running_loss += loss.item()# loss.item () takes the scalar value held in loss.
        
               
                #validation 
                if ind == steps_per_epoch-1:
                    
                    train_losses.append(running_loss)
                    print('training loss on epoch {}: {}'.format(epoch, running_loss))
                    val_loss = 0
                    for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                        batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                        batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                        batch_label= torch.argmax(batch_label, dim=1)
                        outputs = model(batch_input)
                        loss = criterion(outputs, batch_label)
                        val_loss += loss.item()
                        val_losses.append(val_loss)
                        break
        
                    print('validation loss on epoch {}: {}'.format(epoch, val_loss))
                    
        torch.save(model,model_save)
        print("Model Saved!")
        
        
        def draw_graph(val_losses,train_losses,epochs):
            norm_validation = [float(i)/sum(val_losses) for i in val_losses]
            norm_train = [float(i)/sum(train_losses) for i in train_losses]
            epoch_numbers=list(range(1,epochs+1,1))
            plt.figure(figsize=(12,6))
            plt.subplot(2, 2, 1)
            plt.plot(epoch_numbers,norm_validation,color="red") 
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
            plt.title('Train losses')
            plt.subplot(2, 2, 2)
            plt.plot(epoch_numbers,norm_train,color="blue")
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
            plt.title('Validation losses')
            plt.subplot(2, 1, 2)
            plt.plot(epoch_numbers,norm_validation, 'r-',color="red")
            plt.plot(epoch_numbers,norm_train, 'r-',color="blue")
            plt.legend(['w=1','w=2'])
            plt.title('Train and Validation Losses')
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
            
            
            plt.show()
        
        draw_graph(val_losses,train_losses,epochs)
    
# #########POLYGON###############
# ######### PARAMETERS ##########
# valid_size = 0.3
# test_size  = 0.1
# batch_size = 4
# epochs = 20
# cuda = True
# input_shape = (224, 224)
# n_classes = 4
# MASK_DIR=MASK_POLYGON_DIR
# model=POL_NET(input_shape,n_classes)
# model_save='/mnt/sdb2/Intern_2/models/best_polygon_model.pt'
#train_if=True
# ############################### 
# #polygon_model
# train(valid_size,test_size,batch_size,epochs,cuda,input_shape,n_classes,MASK_DIR,model,model_save)



#########LINE###############
######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.05
batch_size = 4
epochs = 20
cuda = True
input_shape = (224, 224)
n_classes = 3
MASK_DIR=MASK_LINE_DIR
model=SegNet(n_classes)
model_save='/mnt/sdb2/Intern_2/models/best_line_model.pt'
train_if=False
############################## 
#polygon_model
train(valid_size,test_size,batch_size,epochs,cuda,input_shape,n_classes,MASK_DIR,model,model_save,train_if)

