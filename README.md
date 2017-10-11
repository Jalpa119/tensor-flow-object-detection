# Goal:
Goal here is building a pipeline that can take the training data and functions to train the model and perform predictions.

# Introduction to other models

This is a repo for implementing object detection with pre-trained models (as shown below) on tensorflow.

| Model name  | Speed | COCO mAP | Outputs |
| ------------ | :--------------: | :--------------: | :-------------: |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz) | fast | 21 | Boxes |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz) | fast | 24 | Boxes |
| [rfcn_resnet101_coco](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz)  | medium | 30 | Boxes |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz) | medium | 32 | Boxes |
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz) | slow | 37 | Boxes |

# Tensor flow object detection steps:
* Step1: Download pre-requisites for tensor-flow and install tensor-flow [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
* Step2: Download pre-trained model from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz)
* Step3: Labeled images, for training and test data, are residing @ tensorflow-object-detection\models\research\object_detection\images
* Step4: Training and test data set was created using helper function 
   1. xml_to_csv.py will create train and test csvs @ \tensorflow-object-detection\models\research\object_detection\data
   ```
   cd \tensorflow-object-detection\
   python xml_to_csv.py
   ```
   2. generate_tf_record.py to create record files. 
   ```
   cd \tensorflow-object-detection\
   - Create train data: (make sure you are pointing to right csv_input and output direcotries.)
    python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record

  - Create test data: (make sure you are pointing to right csv_input and output direcotries.) 
    python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
   ```
* Step5: We are using ssd_mobilenet_v1_coco_11_06_2017 pre-trained model. Download the config [file](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config)
and save it @ tensorflow-object-detection\models\research\object_detection\training

* Stepe6: We can configure batchsize and number of classes, according to our hardware configuration. Also replace the variable `PATH_TO_BE_CONFIGURED` in the config file with `ssd_mobilenet_v1_coco_11_06_2017` also update the train and test.record to point to right directory - in oour case its under `data/` directory.

also set the `label_map_path` to point to our dor pbtxt file, under \tensorflow-object-detection\models\research\object_detection\training\object-detection.pbtxt,  which contains the following.
```
item {
  id: 1
  name: 'pizza'
  }
```
if we have more than one labels then we can continue assigning more ids with name values. 
* Step 7: Train the model
```
cd \tensorflow-object-detection\models\research\object_detection
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```
