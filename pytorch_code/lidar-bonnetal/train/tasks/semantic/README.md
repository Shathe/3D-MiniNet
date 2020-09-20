# 3D-MiniNet: Pytorch Implementation 

This code allows to reproduce the experiments from the 3D-MiniNet paper.

## Dependencies

First you need to install the nvidia driver and CUDA, so have fun!

- CUDA Installation guide: [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

- System dependencies:

  ```sh
  $ sudo apt-get update 
  $ sudo apt-get install -yqq  build-essential ninja-build \
    python3-dev python3-pip apt-utils curl git cmake unzip autoconf autogen \
    libtool mlocate zlib1g-dev python3-numpy python3-wheel wget \
    software-properties-common openjdk-8-jdk libpng-dev  \
    libxft-dev ffmpeg python3-pyqt5.qtopengl
  $ sudo updatedb
  ```

- Python dependencies

  ```sh
  $ sudo pip3 install -r requirements.txt
  ```



## Dataset

Download the SemanticKitti dataset [here](http://semantic-kitti.org/dataset.html#download)

## Configuration files

Architecture configuration files are located at [config/arch](config/arch/)
Dataset configuration files are located at [config/labels](config/labels/)

## Apps

`ALL SCRIPTS CAN BE INVOKED WITH -h TO GET EXTRA HELP ON HOW TO RUN THEM`

### Visualization

To visualize the data (in this example sequence 00):

```sh
$ ./visualize.py -d /path/to/dataset/ -s 00
```

To visualize the predictions (in this example sequence 00):

```sh
$ ./visualize.py -d /path/to/dataset/ -p /path/to/predictions/ -s 00
```

### Training

To train a network (from scratch):

```sh
$ ./train.py -d /path/to/dataset -ac config/arch/CHOICE.yaml -l /path/to/log
```

To train a network (from pretrained model):

```
$ ./train.py -d /path/to/dataset -ac config/arch/CHOICE.yaml -dc config/labels/CHOICE.yaml -l /path/to/log -p /path/to/pretrained
```

Change any training parameters like the number of training epochs in the config/labels/CHOICE.yaml configuration file.

Besides, for a minor boost of performance you can:
- Play with the hyperparameters 
- Train more epochs or doing several trainings lowering the learning rate
- Change the [data augmentation](https://github.com/Shathe/3D-MiniNet-private/blob/master/pytorch_code/lidar-bonnetal/train/common/laserscan.py#L85)
- Add the validation set to the training: line 56 of the (modules/trainer.py): train_sequences=self.DATA["split"]["train"] + self.DATA["split"]["valid"],


### Inference

To infer the predictions for the entire dataset:

```sh
$ ./infer.py -d /path/to/dataset/ -l /path/for/predictions -m /path/to/model
````

### Evaluation

Upload test inferences on the [evaluation platform](https://competitions.codalab.org/competitions/20331)

For validating against the validation set:
```sh
$ ./evaluate_iou.py -d /path/to/dataset/ -p /path/for/predictions --split valid
````



## Pre-trained Models

- [3D-MiniNet](models/3D-MiniNet)
- [3D-MiniNet-small](models/3D-MiniNet-small)
- [3D-MiniNet-tiny](models/3D-MiniNet-tiny)

These models have been trained both on train and validation data.

To enable kNN post-processing, just change the boolean value to `True` in the `arch_cfg.yaml` file parameter, inside the model directory.

### Troubleshooting
- Please, **if you have a CUDA out memory error **lower the batch size, although performance can be compromised.**

- If you have some problems with the GPU (like you have only 1 gpu and it is not founding it correctly) you may add at the beginning of the training file:
``` 
n_gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)
```