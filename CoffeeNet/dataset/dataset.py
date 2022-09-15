import torch
import cv2 as cv
import numpy as np
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms.functional as TF

class CoffeeDataset(Dataset):
    def __init__(self,dataframe,annotation_info,transform=None) -> None:
        super().__init__()
        self.dataframe = dataframe
        self.annotation_info = annotation_info
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        # get the file names from the dataframe
        image_path = self.dataframe.iloc[index,0]
        mask_path = self.dataframe.iloc[index,1]
        # read the image and mask
        image = cv.imread(image_path)
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        mask = cv.imread(mask_path)
        mask = cv.cvtColor(mask,cv.COLOR_BGR2RGB)
        # separate the masks information
        leaf = cv.inRange(mask,np.array(self.annotation_info["background"]),np.array(self.annotation_info["symptom"]))
        # inverse
        leaf = cv.bitwise_not(leaf)
        symptom = cv.inRange(mask,np.array(self.annotation_info["background"]),np.array(self.annotation_info["leaf"]))
        symptom = cv.bitwise_not(symptom)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return(image,mask)