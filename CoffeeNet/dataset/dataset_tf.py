import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
import pandas as pd

class CoffeeDatasetTF(keras.utils.Sequence):
    def __init__(self,dataframe:pd.DataFrame,batch_size:int,img_size:tuple,shuffle:bool=False):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.dataframe) // self.batch_size
    
    def __getitem__(self,idx):
        """ Generate one batch of data """
        batches = self.dataframe.iloc[idx * self.batch_size : (idx+1)*self.batch_size]
        # get the batch of input images
        batch_images = batches.iloc[:,0]
        # get the batch of mask images
        batch_masks = batches.iloc[:,1]

        X = np.asarray([self._read_img(img) for img in batch_images])
        Y = np.asarray([self._read_img(img) for img in batch_masks])

        return X.astype(np.float32),Y.astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def _read_img(self,img):
        image = cv.imread(img)
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = cv.resize(image, self.img_size, interpolation = cv.INTER_AREA)
        return image/255.0