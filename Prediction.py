# System Modules

import matplotlib.pyplot as plt
import skvideo.io

# Deep Learning Modules
from torch.utils.data import Dataset

# User Defined Modules
from torch.nn import *
from serde import read_config
from utils.visualization import *

class Prediction:
    '''
    This class represents prediction process similar to the Training class.

    '''
    def __init__(self,cfg_path):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.model_info=self.params['Network']
        self.model_info['seed']=self.model_info['seed']
        self.setup_cuda()
        

        
    def setup_cuda(self, cuda_device_id=0):
        '''Setup the CUDA device'''
        #Refer similar function from training
        torch.backends.cudnn.fastest = True
        torch.cuda.set_device(cuda_device_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.manual_seed_all(self.model_info['seed'])
        torch.manual_seed(self.model_info['seed'])

        
    def setup_model(self, model,model_file_name=None):
        '''
        Setup the model by defining the model, load the model from the pth file saved during training.
        
        '''
        # Use the default file from params['trained_model_name'] if 
        # user has not specified any pth file in model_file_name argument
        if model_file_name == None:
            model_file_name = self.params['trained_model_name'];
        #Set model to self.device
        self.model = model().to(self.device)
        net_path = self.params["network_output_path"]
        model_path=os.path.join(net_path,model_file_name);
        #Load model from model_file_name and default network_output_path
        self.model.load_state_dict(torch.load(model_path))
        
    def predict(self,predict_data_loader,visualize=True,save_video=False):
        # Read params to check if any params have been changed by user
        self.params = read_config(self.cfg_path)
        #Set model to evaluation mode
        self.model.eval();
        
        if save_video:
            self.create_video_writer()
            

        with torch.no_grad():
            for j, images in enumerate(predict_data_loader):
                #Batch operation: depending on batch size more than one image can be processed.
                #Set images to self.device
                images = images.to(self.device);
                #Provide the images as input to the model and save the result in outputs variable.
                outputs = self.model(images);
               
                #print images.size()
            #for each image in batch
                for i in range(outputs.size(0)):
                    #print(i);
                    image=images[i]/255
                    output=outputs[i]
                    overlay_img,prob_img=get_output_images(image,output);
                    plt.figure(figsize=(10, 8));
                    plt.subplot(1,3,1);
                    plt.imshow(overlay_img);
                    plt.subplot(1,3,2);
                    plt.imshow(prob_img.detach().cpu().numpy());
                    plt.subplot(1,3,3);
                    prob_img = prob_img > 0.5
                    plt.imshow(prob_img.detach().cpu().numpy());
                    plt.show();

                    
                    #Get overlay image and probability map using function from utils.visualization
                

                    #Convert image, overlay image and probability image to numpy so that it can be visualized using matplotlib functions later. Use convert_tensor_to_numpy function from below.



                    #if save_video:
                        #Concatentate input and overlay image(along the width axis [axis=1]) to create video frame. Hint:Use concatenate function from numpy
                        #Write video frame


                    #if(visualize):
                        #display_output(image,prob_img,overlay_img)

            if save_video:
                self.writer.close()
                #Uncomment the below line and replace ??? with the appropriate filename
                #return play_notebook_video(self.params['output_data_path'],???)
            
            
    def create_video_writer(self):
        '''Initialize the video writer'''
        filename="outputvideo.webm"
        output_path=self.params['output_data_path']
        self.writer = skvideo.io.FFmpegWriter(os.path.join(output_path, filename).encode('ascii'),
                                              outputdict={'-vcodec':'libvpx-vp9','-r':'25','-pix_fmt':'yuv420p','-quality':'good','-speed':'2','-threads':'6'},
                                             )
       

    def convert_tensor_to_numpy(self,tensor_img):
        '''
        Convert the tensor image to a numpy image

        '''
        #torch has numpy function but it requires the device to be cpu
        np_img = img.transpose((1, 2, 0));

        # np_img image is now in  C X H X W
        # transpose the array to H x W x C
                

        return np_img
        
        
    

        


        
        
