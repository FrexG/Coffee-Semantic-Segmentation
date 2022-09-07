from __future__ import annotations
import os
import numpy as np
import cv2 as cv
import json
from tqdm import tqdm

class CropAndSave:
    # Initialize data paths
    SOURCE_DATA_PATH = "../data/external"
    DESTINATION_DATA_PATH = "../data/processed"
    
    def __init__(self,data_split = "train") -> None:
        self.data_split = data_split
        # Change the working directory to the current folder
        if not os.path.exists(self.SOURCE_DATA_PATH):
            raise FileNotFoundError
        if not os.path.exists(self.DESTINATION_DATA_PATH):
            os.mkdir(self.SOURCE_DATA_PATH)
        # read images and annotations
        self.images,self.annotations = self.__read_image_and_annotations(data_split)

    def __read_image_and_annotations(self,data_split):
        image_path = os.path.join(self.SOURCE_DATA_PATH,f"images/{data_split}")
        annotation_path = os.path.join(self.SOURCE_DATA_PATH,f"annotations/{data_split}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError
        if not os.path.exists(annotation_path):
            raise FileNotFoundError
        # Get the training image lists
        images = os.listdir(image_path)
        images = sorted(images)
        # Get the annotation image list
        annotations = [annotation_path+"/"+image.split('.')[0]+"_mask.png" for image in images]
        # Append full path to images
        images = [image_path+"/"+image for image in images]
        
        return images,annotations

    def crop_and_save(self,destination_path = None):
        """
            A Function that reads images and annotations, crop the images and annotations
            to the dimension of the leaf and write the croped image to the destination path

            Parameters
            ----------
            destination_path : str
                destination path of the cropped images
            Returns
            -------
                None
        """
        # If destination path is not given then set to default
        if destination_path == None:
                destination_path = self.DESTINATION_DATA_PATH

        if os.path.exists(destination_path):
            out_image_path  = os.path.join(destination_path,f'images/{self.data_split}')
            out_annotation_path  = os.path.join(destination_path,f'annotations/{self.data_split}')
            
            if os.path.exists(out_image_path):
                print(f"Directory '{out_image_path}' exists")
            else:
                print(f"Creating Directory '{out_image_path}'")
                # Create the folder
                os.mkdir(out_image_path)

            if os.path.exists(out_annotation_path):
                print(f"Directory '{out_annotation_path}' exists")
            else:
                print(f"Creating Directory '{out_annotation_path}'")
                # Create the folder
                os.mkdir(out_annotation_path)

            # Read annoation file
            with open(f'{self.SOURCE_DATA_PATH}/annotations-info.json') as f:
                annotation_info = json.load(f)
                annotation_info
            ## Loop through all the images
            for i,image in enumerate(tqdm(self.images)):
                image_file = cv.imread(image)
                annotation_file = cv.imread(self.annotations[i])
                # Convert image and annotation to RGB
                image_file_rgb = cv.cvtColor(image_file,cv.COLOR_BGR2RGB)
                annotation_file_rgb = cv.cvtColor(annotation_file,cv.COLOR_BGR2RGB)
                # Separate the leaf mask from the annotation file
                leaf_mask = cv.inRange(annotation_file_rgb,np.array(annotation_info["background"]),np.array(annotation_info["symptom"]))
                leaf_mask = cv.bitwise_not(leaf_mask)
                # Get the cropped image
                cropped_image,cropped_annotation = self.crop(image_file,annotation_file,leaf_mask)

                # Write the cropped image and annotation to the destination
                cv.imwrite(os.path.join(out_image_path,image.split('/')[-1]),cropped_image)
                cv.imwrite(os.path.join(out_annotation_path,self.annotations[i].split('/')[-1]),cropped_annotation)
                # remove the if block below to resize all images
                if i == 2:
                    break

    def crop(self,image,annotation,mask):
        # find the contours from the mask
        contours,heirarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        # find the largest contour
        largest_contour = max(contours,key=cv.contourArea)
        # get the bounding rectangle of the largest contour
        x,y,w,h = cv.boundingRect(largest_contour)
        # crop the image and the mask
        image_cropped = image[y:y+h,x:x+w]
        annotation_cropped = annotation[y:y+h,x:x+w]

        return image_cropped,annotation_cropped

if __name__ == "__main__":
    c = CropAndSave()
    c.crop_and_save()

