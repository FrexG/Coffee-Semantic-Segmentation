import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
""" 
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path) """
    
from .unet_parts_tf import DoubleConv,Down,Up

class Unet(Model):
    def __init__(self,num_classes,num_channels):
        super(Unet,self).__init__()
        # Input
        self.inc = DoubleConv(num_channels,64)
        # Encoder
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,1024)
        # Decoder
        self.up1 = Up(1024,512)
        self.up2 = Up(512,256)
        self.up3 = Up(256,128)
        self.up4 = Up(128,64)
        # output conv
        self.out_conv = keras.layers.Conv2D(3,1,padding="same")

    def call(self,inputs):
        x = self.inc(inputs)
        down1_feature = self.down1(x)
        down2_feature = self.down2(down1_feature)
        down3_feature = self.down3(down2_feature)
        down4_feature = self.down4(down3_feature)
        
        up1_feature = self.up1(down4_feature,down3_feature)
        up2_feature = self.up2(up1_feature,down2_feature) 
        up3_feature = self.up3(up2_feature,down1_feature)
        up4_feature = self.up4(up3_feature,x)  

        x = self.out_conv(up4_feature)

        return tf.nn.softmax(x)

if __name__ == "__main__":
    m = Unet(3,3)
    m.build(input_shape=(1,128,128,3))