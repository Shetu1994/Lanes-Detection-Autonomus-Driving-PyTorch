#UNet Implementation
#Refer to the block diagram and build the UNet Model
#You would notice many blocks are repeating with only changes in paramaters like input and output channels
#Make use of the remaining classes to construct these repeating blocks. You may follow the order of ToDo specified
#above each class while writing the code.


#Additional Task: 
#How are wieghts inintialized in this model?
#Read about the various inintializers available in pytorch
#Define a function to inintialize weights in your model
#and create experiments using different initializers.
#Set the name of your experiment accordingly as 
#this initializer information will not be available
#in config file for later reference.
#You can also implement some parts of this task in other
#scripts like Training.py


import torch
import torch.nn as nn
import torch.nn.functional as F

#ToDo 5
class UNet(nn.Module):
    def __init__(self, n_in_channels=3, n_out_classes=2):
        super(UNet, self).__init__()
        #Create object for the components of the network. You may also make use of inconv,up,down and outconv classes defined below.

    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 

        return output_tensor
#ToDo 1: Implement double convolution components that you see repeating throughout the architecture.
class double_conv(nn.Module):
    #(conv => Batch Normalization => ReLU) * 2
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        #Create object for the components of the block



    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor  

        return output_tensor

#ToDo 2: Implement input block
class inconv(nn.Module):
    #Input Block
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        #Create object for the components of the block. You may also make use of double_conv defined above.


    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 

        
        return output_tensor

#ToDo 2: Implement generic down block
class down(nn.Module):
    #Down Block
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        #Create object for the components of the block.You may also make use of double_conv defined above.


    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 


        return output_tensor

#ToDo 3: Implement generic up block
class up(nn.Module):
    #Up Block
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        # Create an object for the upsampling operation
        
        #Create an object for the remaining components of the block.You may also make use of double_conv defined above.
        



    def forward(self, input_tensor_1, input_tensor_2):
        #Upsample the input_tensor_1

        #Concatenation of the  upsampled result and input_tensor_2


        #Apply concatenated result to the object containing remaining components of the block and return result


        return output_tensor

#ToDo 4: Implement out block
class outconv(nn.Module):
    #Out Block
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        #Create object for the components of the block


    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 

        return output_tensor
