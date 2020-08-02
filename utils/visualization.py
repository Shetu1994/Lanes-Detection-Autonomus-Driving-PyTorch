import matplotlib.pyplot as plt
from IPython.display import Markdown as md
from torchvision import transforms
import torch
from PIL import Image
import PIL
import torchvision
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_probability_map(output_3d):
    '''Returns the probabilty map tensor(2d: H * W) for the given network-output tensor (3d:C* H * W)'''
    #Use softmax
    out_4d = torch.unsqueeze(output_3d, 0)
    gpm = torch.nn.Softmax2d()
    output = gpm(out_4d)
    
    return output[0][1] #Returns the 2nd dimenson because 1st dimension is pointless
    


def get_output_images(input_layer,output_3d):
    '''Returns overlayed image and probabilty map in 3d: C * H * W'''
    # Generate the probabilty map using above funtion and 
    # create a 3 channel image with probabilty map in the red channel.
    #ch,row,col= input_layer.shape;
    #overlay_image = np.zeros([ch,row,col])
    overlay_image = input_layer;
    prob_image = get_probability_map(output_3d)
    #print prob_image
    prob_image1 = prob_image > 0.5
    #print prob_image
    overlay_image[0,:,:] = overlay_image[0,:,:] + prob_image1.float();
    
    #Use round/threshold function to divide pixels to each class
    
    #Overlay the class image on top of the input image to give a 
    #semi transperant red at pixels where class value =1 or lane 
    # You may use other modules to create the images but 
    #the return value must be 3d tensors

    overlay_image = np.transpose(overlay_image,[1,2,0])

    return overlay_image,prob_image



def play_notebook_video(folder,filename):
    '''Plays video in ipython using markdown'''
    file_path=os.path.join(folder, filename)  
    return md('<video controls src="{0}"/>'.format(file_path))




def display_output(image,prob_img,overlay_img):
    '''
    Displays the output using matplotlib subplots
    
    '''
    #Inputs are numpy array images.
    plt.figure();
    plt.subplot(1,3,1);
    plt.imshow(overlay_img);
    plt.subplot(1,3,2);
    plt.imshow(prob_img);
    plt.subplot(1,3,3);
    plt.imshow(prob_img);
    plt.show();


    
 



    
    
