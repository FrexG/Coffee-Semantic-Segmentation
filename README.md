# CoffeeNet 2.0
Coffee leaf semantic segmentation and disease severity estimation based on UNet and DeepLabv3 architectures
## Description
A deep learning approach for the detection and severity estimation of coffee arabica leaf rust disease.
This work is a continuation of our previous work titled [@CoffeeNetV1](https://github.com/FrexG/AIC_Coffee_Disease_DL) where we utilised 
various image processing [@algorithms](https://onlinelibrary.wiley.com/doi/full/10.1002/int.22747) to perform segmentation of background 
and leaf regions.
Here, we utilize deep learning semantic segmentation architectures such as UNet and DeepLabv3 to perform segmentation of regions and calculate 
the disease severity based on the defined metrics.
#### This work is currently on progress!!!


## Getting Started

### Dependencies

* Python >3.9.X
* OpenCV >4.x.x
* torch >1.2.x
* torchvision >0.13.x
* numpy >1.22.x
* tensorflow >2.x

## Help
* View the main branch for pytorch implementation
* Tensorflow brach for the tensorflow implementation

## License

This project is licensed under the [GNU General Public License v3.0] License - see the LICENSE.md file for details

## Acknowledgments

The original data used on this work is obtained from [here](https://github.com/esgario/lara2018) and was used by the authors
of [Esgario et al.](https://www.sciencedirect.com/science/article/abs/pii/S0168169919313225) for their research work.