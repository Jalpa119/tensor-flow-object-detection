# tensor-flow-object-detection
## Description
*Step1: Download pre-requisites for tensor-flow and install tensor-flow(https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
*Step2: Download pre-trained model from here(http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz)

# Introduction to other models

This is a repo for implementing object detection with pre-trained models (as shown below) on tensorflow.

| Model name  | Speed | COCO mAP | Outputs |
| ------------ | :--------------: | :--------------: | :-------------: |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz) | fast | 21 | Boxes |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz) | fast | 24 | Boxes |
| [rfcn_resnet101_coco](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz)  | medium | 30 | Boxes |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz) | medium | 32 | Boxes |
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz) | slow | 37 | Boxes |



# Run Demo


```
# Clone this repo
git clone https://github.com/KleinYuan/tf-object-detection.git

# Setting up
cd tf-object-detection
bash setup.sh

# Run demo


```
