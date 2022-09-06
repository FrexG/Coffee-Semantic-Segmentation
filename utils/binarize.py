import os
import cv2 as cv
import matplotlib.pyplot as plt

base_directory = '/home/frexg/Downloads/lara2018-master/segmentation/dataset'

annotations_directory = os.path.join(base_directory, 'annotations')
images_directory = os.path.join(base_directory, 'images')
output_directory = os.path.join(base_directory, 'validation_binary_sens')

for dirpath, dirnames, filename in os.walk(f'{annotations_directory}/val'):
    for im in filename:
        image = cv.imread(f'{dirpath}/{im}')
        mask = cv.inRange(image, (0, 0, 0), (0, 176, 0))
        mask = cv.bitwise_not(mask)
        cv.imwrite(os.path.join(output_directory, im), mask)
