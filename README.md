# Facial Kinship Verification using Deep Learning

Manuel Rios & Tomas Correa

This folder contains the development of final project for the Computer Vision course at Universidad de los Andes.


## Dataset 

Experiments are configured to run smoothly on **BCV002 machine**. Dataset is available in the following path:  
*media/disk0/Datasets_FP/Correa_Rios*. Throughout a symbolic link, all permissions are granted.

All execution modes will use our best method FaceNet with triplet and classification loss.

## Training:

All default arguments can be found inside **main.py**.  
To train run the following command:

python main.py --mode train

Optional arguments are learning rate, number of epochs, and batch size:

python main.py --mode train --lr 0.1 --num_epochs 16 --bs 48
 

## Evaluation

Testing should be done with the following command:

python main.py --mode test


## Demo

Demo should be done with the following command:

python main.py --mode demo --pair_type md

Where the pair_type argument indicates the kinship relation to validate (choices are fs, fd, ms, md).

Optional arguments are the test images to use, requiring only to specify the test image file name (not whole path):

python main.py --mode demo --pair_type md --img1 face1.jpg --img2 face2.jpg