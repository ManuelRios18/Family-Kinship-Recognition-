from trainers.kinship_trainer import KinshipTrainer
from PIL import Image
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Kinship Verification.')

parser.add_argument('--data_path', type=str, default=r"/media/disk0/Datasets_FP/Correa_Rios",
                    help='path to dataset')
parser.add_argument('--model_name', type=str, default='kin_facenet',
                    help='name of learning model')
parser.add_argument('--dataset_name', type=str, default='fiw',
                    help='name of dataset')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu id to use')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--bs', type=int, default=40,
                    help='batch size')
parser.add_argument('--num_epochs', type=int, default=16,
                    help='number of training epochs')
parser.add_argument('--mode', type=str, default='demo',
                    help='mode to use',choices=('train','test','demo'))
parser.add_argument('--img1', type=str, default='suegra_manuel_crop.jpg',
                    help='parent image for demo')
parser.add_argument('--img2', type=str, default='hijo_manuel_crop.jpg',
                    help='child imge for demo')
parser.add_argument('--pair_type', type=str, default='ms',
                    help='pair type for demo',choices=('fd','fs','md','ms'))


args = parser.parse_args()

# model_name = "vgg_siamese"
exp_sufix = args.model_name + "_with_norm_"

momentum = 0.9
weight_decay = 0.005
optimizer_name = "SGD"
target_metric = "acc"

# dataset_path = "/home/msrios/kinfacew/"
# dataset_path = "/media/manuel/New Volume/Computer Vision/fiw"
args.data_path = "..\..\Data\FIW"
kin_pairs = ["fd", "ms", "md", "fs"]
kinfacew_set_name = "KinFaceW-II"
kinfacew_n_folds = 5

vgg_path = "/home/msrios/vgg_weights/VGG_FACE.t7"

trainer = KinshipTrainer(model_name=args.model_name, optimizer_name=optimizer_name, lr=args.lr, momentum=momentum,
                         weight_decay=weight_decay,  n_epochs=args.num_epochs, dataset=args.dataset_name, dataset_path=args.data_path,
                         kin_pairs=kin_pairs, batch_size=args.bs, exp_sufix=exp_sufix, gpu_id=args.gpu,
                         kinfacew_set_name=kinfacew_set_name, kinfacew_n_folds=kinfacew_n_folds,
                         target_metric=target_metric, vgg_weights=vgg_path)

if args.mode=='train':
    trainer.train()
if args.mode=='test':
    trainer.test_fiw()
if args.mode=='demo':
    trainer.demo(args.img1,args.img2,args.pair_type)
