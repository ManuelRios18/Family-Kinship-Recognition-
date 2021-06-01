# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 01:52:56 2021

@author: tomco
"""


from trainers.kinship_trainer import KinshipTrainer
from PIL import Image
import numpy as np
import os


model_name = "kin_facenet"
# model_name = "vgg_siamese"
exp_sufix = model_name + "_with_norm_"

n_epochs = 75
lr = 0.001
batch_size = 40
momentum = 0.9
weight_decay = 0.005
gpu_id = 0
optimizer_name = "SGD"
target_metric = "acc"

# dataset = "kinfacew"
dataset = "fiw"
# dataset_path = "/home/msrios/kinfacew/"
# dataset_path = "/media/manuel/New Volume/Computer Vision/fiw"
dataset_path = "..\..\Data\FIW"
kin_pairs = ["fd", "ms", "md", "fs"]
kinfacew_set_name = "KinFaceW-II"
kinfacew_n_folds = 5



tester = KinshipTrainer(model_name=model_name, optimizer_name=optimizer_name, lr=lr, momentum=momentum,
                         weight_decay=weight_decay,  n_epochs=n_epochs, dataset=dataset, dataset_path=dataset_path,
                         kin_pairs=kin_pairs, batch_size=batch_size, exp_sufix=exp_sufix, gpu_id=gpu_id,
                         kinfacew_set_name=kinfacew_set_name, kinfacew_n_folds=kinfacew_n_folds,
                         target_metric=target_metric)

parent_path = os.path.join(dataset_path,'test-faces','face367.jpg')
child_path = os.path.join(dataset_path,'test-faces','face405.jpg')

parent_image = np.asarray(Image.open(parent_path))
child_image = np.asarray(Image.open(child_path))

# parent_image = np.array([parent_image,parent_image])
# child_image = np.array([child_image,child_image])

tester.demo(parent_image,child_image,'md')