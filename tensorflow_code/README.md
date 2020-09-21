# 3D-MiniNet: Tensorflow Implementation

The original code is the one implemented in pytorch. This is just a port to tensorflow.
## Requirements
The instructions have been tested on Ubuntu 16.04 with Python 3.6 and Tensorflow 1.13 with GPU support.

Download the SqueezeSeg dataset as explained [here](https://github.com/xuanyuzhou98/SqueezeSegV2) in Section "Dataset". Then, in **make_tfrecord_pn.py** line 29, set the variable "**semantic_base**" to the path of the SqueezeSeg dataset folder.

```python
semantic_base = "/path/to/squeezeseg/dataset/"
```

## Train and validate 3D-MiniNet
You can generate the _training_ and _validation_ **TFRecords** by running the following command:
```bash
python make_tfrecord_pn.py --config=config/3dmininet.cfg 
```

Once done, you can start training the model by running:
```bash
python train.py --config=config/3dmininet.cfg --gpu=0
```
You can set which GPU is being used by tuning the `--gpu` parameter.

While training, you can run the validation as following:
```bash
python test.py --config=config/3dmininet.cfg --gpu=1
```
This script will run the validation on each new checkpoint as soon as they are created, and will store the scores in a text file.

During training, the code will produce logs for Tensorboard, as specified in the configuration file (see after). You can run Tensorboard while training with the following command:
```bash
tensorboard --logdir=training_path/logs
```

## Data Augmentation 

The current data augmentation is enable. You can disable commenting [this line](https://github.com/Shathe/3D-MiniNet/blob/master/tensorflow_code/train.py#L268).
The data augmentation explained in the paper is for a 360º view LIDAR and this dataset is not.
For this dataset other data augmentation was implemented see the data [augmentation file](augmentation.py).
For enabling the data augmentation just uncomment the line 268 from the training file.
**The data augmentation is specific for this dataset and dataset format so if you are using other dataset, do not use this one.**
Besides, the current daat augmentation is done in the 2D space (images), not on the 3D space (point cloud) which is preferable.
