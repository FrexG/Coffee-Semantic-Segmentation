import tensorflow as tf
from tensorflow import keras
# Construct unet parts
class DoubleConv(keras.layers.Layer):
    """(Conv2D => BN => ReLu * 2"""
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()

        self.conv1 = keras.layers.Conv2D(filters=in_channels,kernel_size=(3,3),
        padding="same",use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.relu1 = keras.layers.ReLU()

        self.conv2 = keras.layers.Conv2D(filters=out_channels,kernel_size=(3,3),
        padding="same",use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()
        self.relu2 = keras.layers.ReLU()

    def call(self,inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class Down(keras.layers.Layer):
    """ Downscaling with maxpoll then double conv"""
    def __init__(self,in_channels,out_channels) -> None:
        super(Down,self).__init__()
        self.maxpool = keras.layers.MaxPooling2D((2,2))
        self.double_conv = DoubleConv(in_channels,out_channels)

    def call(self,inputs):
        x = self.maxpool(inputs)
        x = self.double_conv(x)
        return x

class Up(keras.layers.Layer):
    """ Upscalign then double conv """
    def __init__(self,in_channels,out_channels):
        super(Up,self).__init__()
        self.up = keras.layers.Conv2DTranspose(out_channels,3,2,padding="same")
        self.double_conv = DoubleConv(in_channels,out_channels)
    def call(self,inputs,conv_features):
        x = self.up(inputs)
        # Concatenate
        x = keras.layers.concatenate([x,conv_features])
        return self.double_conv(x)
        