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
        #Create object for the components of the network. You may also make use of inconv,up,down and outconv classes defined below.self.inc = inconv(n_channels, 64)
        self.inc = inconv(n_in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_out_classes)
        
    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor
        x1 = self.inc(input_tensor)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        output_tensor = self.up1(x5, x4)
        output_tensor = self.up2(output_tensor, x3)
        output_tensor = self.up3(output_tensor, x2)
        output_tensor = self.up4(output_tensor, x1)
        output_tensor = self.outc(output_tensor)
        return output_tensor
    
#ToDo 1: Implement double convolution components that you see repeating throughout the architecture.
class double_conv(nn.Module):
    #(conv => Batch Normalization => ReLU) * 2
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        #Create object for the components of the block
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor  
        output_tensor = self.conv(input_tensor)
        return output_tensor

#ToDo 2: Implement input block
class inconv(nn.Module):
    #Input Block
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        #Create object for the components of the block. You may also make use of double_conv defined above.
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        output_tensor = self.conv(input_tensor)
        return output_tensor

#ToDo 2: Implement generic down block
class down(nn.Module):
    #Down Block
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        #Create object for the components of the block.You may also make use of double_conv defined above.
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        output_tensor = self.mpconv(input_tensor)
        return output_tensor

#ToDo 3: Implement generic up block
class up(nn.Module):
    #Up Block
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        # Create an object for the upsampling operation
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #Create an object for the remaining components of the block.You may also make use of double_conv defined above.
        self.conv = double_conv(in_ch, out_ch)
        



    def forward(self, input_tensor_1, input_tensor_2):
        #Upsample the input_tensor_1
        input_tensor_1 = self.up(input_tensor_1)
        
        # input is CHW
        diffY = input_tensor_2.size()[2] - input_tensor_1.size()[2]
        diffX = input_tensor_2.size()[3] - input_tensor_1.size()[3]

        input_tensor_1 = F.pad(input_tensor_1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        

        output_tensor = torch.cat([input_tensor_2, input_tensor_1], dim=1)
        output_tensor = self.conv(output_tensor)
        #Concatenation of the  upsampled result and input_tensor_2
        #Apply concatenated result to the object containing remaining components of the block and return result
        return output_tensor

#ToDo 4: Implement out block
class outconv(nn.Module):
    #Out Block
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        #Create object for the components of the block
        self.outconv = nn.Conv2d(in_ch, out_ch, 1);

    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        output_tensor = self.outconv(input_tensor);
        return output_tensor
