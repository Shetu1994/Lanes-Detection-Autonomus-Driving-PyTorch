from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte


from torch.utils.data import Dataset
import torch

from specs import *
from serde import read_detection, read_config,write_config
from pipelines import simulation_pipeline
from Training import Mode
from pipelines import label_true_negatives
from config import sim_config

sc = sim_config()['simulator']

HEIGHT=sc['height']
WIDTH=sc['width']
DEFAULT_DATA_PATH=read_config('./config.json')['input_data_path']

class Img_dataset(Dataset):
    """
    Class representing Image Dataset
    This class is used to represent both Simulation datasets and true negative real world image datasets.
    The images are also returned in the HEIGHT and WIDTH obtained from the simulator config.
    Depending on the mode specified by the user, the Dataset return labels for train and test modes.
    User also has the option of choosing smaller sample from a folder containg large number of images by setting the size 
    parameter
    
    Refer: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, dataset_name, size,cfg_path, mode=Mode.TRAIN, dataset_parent_path=DEFAULT_DATA_PATH
                 , augmentation=None, seed=1):
        """
        Args:
            dataset_name (string): Folder name of the dataset.
            size (int):
                No of images to be used from the dataset folder
            mode (enumeration Mode):
                Nature of operation to be done with the data.
                Possibe inputs are Mode.PREDICt,Mode.TRAIN,Mode.TEST
                Default value: Mode.TRAIN
            dataset_parent_path (string):
                Path of the folder where the dataset folder is present
                Default: DEFAULT_DATA_PATH as per config.json
            cfg_path (string):
                Config file path of your experiment

            augmentation(Augmentation object):
                Augmentation to be applied on the dataset. Augmentation is passed using the object
                from Compose class (see augmentation.py)
            seed
                Seed used for random functions
                Default:1

        """
        params = read_config(cfg_path)
        self.cfg_path=params['cfg_path']

        self.detections_file_name = params['detections_file_name']

        self.mode = mode
        self.dataset_path=os.path.join(dataset_parent_path,dataset_name)
        self.datset_name=dataset_name
        self.size=size
        self.augmentation = augmentation
        self._init_dataset(dataset_name,seed,params)





    def __len__(self):
        '''Returns length of the dataset'''
        return self.size

    def __getitem__(self, idx):
        '''
        Using self.img_list and the argument value idx, return images and labels(if applicable based on Mode)
        The images and labels are returned in torch tensor format

        '''

        #Read images using files name availble in self.img_list
        img = imread(os.path.join(self.dataset_path, self.img_list[idx]))

        #Resize image if required
        img = resize(img, (HEIGHT,WIDTH, 3),mode='constant')
        img = img_as_ubyte(img)

        #If mode is PREDICT, convert image to tensor and return the image
        if self.mode ==Mode.PREDICT :
            # Convert image to tensor
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).float()
            return img
        
        #If mode is not PREDICT,  and retun image ,label
        #Obtain label from the detection file
        detection = read_detection(
            os.path.join(self.dataset_path, self.detections_file_name),self.img_list[idx])

        #print os.path.join(self.dataset_path, self.detections_file_name);
        y = detection[0]
        x = detection[1]
        label = np.zeros((HEIGHT, WIDTH),dtype=np.uint8)
        label[y, x] = 1
        #Apply augmentation if applicable
        if self.augmentation is not None:
            img, label = self.augmentation(img, label)
        #Convert image and label to tensor and returm image,label
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        label = torch.from_numpy(label).long()

        return img, label


    def _init_dataset(self,dataset_name,seed,params):
        '''
        Initialize Dataset: Get the list the list of images from the dataset.

        If the dataset is found , the size is checked against the number of images in the folder and
        if size is more than number of images in the folder, randomly pick images from the folder so
        that number of images is equal to the user specified size.

        If the dataset is not found, the simulator is run and a folder containing simulation
        images and detection files, is created with the given dataset name.

        Final image list is stored into self.img_list


        '''



        # Check if the directory exists
        if os.path.isdir(self.dataset_path):
            # Collect the file names of all images and store them to a list
            img_list= [name for name in os.listdir(self.dataset_path) if name.endswith(('.jpg', '.jpeg', '.png'))]
            image_count_in_folder=len(img_list)
            #Compare number of images in img_list with size provided by user and then assign img_list to self.img_list

            #If number of images < size: inform user about the available image count
            #and change value of self.size to number of images in img_list
            #assign img_list to self.img_list without changes
            if image_count_in_folder < self.size:
                print('{0}\nImages availabe in folder: \t{1}\nGiven Size: \t\t\t{2}\n'
                      'Images used in dataset:  \t{1}\n\n\n'.format(dataset_name, image_count_in_folder, self.size))
                self.size=len(img_list)
                self.img_list=img_list


            # if number of images >size : inform user about the availabe image count
            #Randomly select images from img_list and  assign them into self.img_list such that self.img_list
            # would contain number of images as specified by user. (Use the seed specified by user for random function)

            elif image_count_in_folder > self.size:
                np.random.seed(seed)
                idx=np.random.choice(image_count_in_folder,self.size)
                self.img_list=[img_list[i] for i in idx]

                print('{0}\nImages availabe in folder: \t{1}\nGiven Size: \t\t\t{2} \n'
                      'Images used in dataset:  \t{2}\n\n\n'.format(dataset_name, image_count_in_folder, self.size))

                # If number of images = size
                # assign img_list to self.img_list without changes
            else:
                 self.img_list=img_list


        else:

            # Run the simulator and save the results in the dataset path
            simulation_pipeline(params, self.size, dataset_name, seed)
            self.img_list= [name for name in os.listdir(self.dataset_path) if name.endswith(('.jpg', '.jpeg', '.png'))]


        #Check if detection file is present in the folder and if it is not present create a true negative(dummy) detection file using
        #label_true_negatives function from pipelines.py

        if not os.path.exists(os.path.join(self.dataset_path,self.detections_file_name)):

            label_true_negatives(self.dataset_path, self.detections_file_name)



        #Save the dataset information for writing to config file
        if self.mode==Mode.TRAIN:
            params = read_config(self.cfg_path)
            params['Network']['total_dataset_number']+=1
            dataset_key='Traing_Dataset_'+str(params['Network']['total_dataset_number'])
            #If augmentation is applied
            if self.augmentation:
                augmenetation_applied=[i.__class__.__name__ for i in self.augmentation.augmentation_list]
            else:
                augmenetation_applied=None
            dataset_info={
                'name':dataset_name,
                'path':self.dataset_path,
                'size':self.size,
                'augmenetation_applied':augmenetation_applied,
                'seed':seed
            }
            params[dataset_key]=dataset_info
            write_config(params, params['cfg_path'],sort_keys=True)

