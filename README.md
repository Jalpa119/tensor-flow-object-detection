# Goal:
Goal here is building a pipeline that can take the training data and functions to train the model and perform predictions.
# Source Code: 
[Download it from here](https://drive.google.com/drive/folders/0ByTKasiHTv3abDZrOHhZMkI0Z00?usp=sharing)
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
* Step 8 : for us, in the models/object_detection directory, there is a script that does this for us: export_inference_graph.py
-> To run this, you just need to pass in your checkpoint and your pipeline config, then wherever you want the inference graph to be placed , In my Case the code is

python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-1238\
    --output_directory pizza_graph

Run the above command in Model/Object_detection and then go to the pizza_graph directory and you will find new directory saved_model and most importantly, the forzen_inference_graph.pb file.

* Step 9 : Go to Model/object_dection/jupyter notebook 
Open the object_detection_tutorial.ipynb (in this we have to do some changes)

# What model to download.
MODEL_NAME = 'pizza_graph'


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 90

--> In the Detection part make some changes in my case:
# For the sake of simplicity we will use only 4 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 6) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

Step 10 : Run All code You will get your answer . 

