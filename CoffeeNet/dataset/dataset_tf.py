import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras

class CoffeeDatasetTF(keras.utils.Sequence):
    def __init__(self,dataframe,annotation_info,batch_size):
        self.dataframe = dataframe
        self.annotation_info = annotation_info
        self.batch_size = batch_size

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

        print(X.shape,Y.shape)
        return X,Y

    def _read_img(self,img):
        image = cv.imread(img)
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = cv.resize(image, (256,256), interpolation = cv.INTER_AREA)
        return image/255.0