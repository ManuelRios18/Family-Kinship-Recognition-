from trainers.kinship_trainer import KinshipTrainer

# model_name = "small_face_model"
model_name = "vgg_siamese"
exp_sufix = model_name

n_epochs = 45
lr = 0.001
batch_size = 8
momentum = 0.9
weight_decay = 0.005
gpu_id = 0
optimizer_name = "SGD"
target_metric = "acc"

dataset = "kinfacew"
# dataset = "fiw"
dataset_path = "/home/manuel/Documents/masters/Computer Vision/YGYME/data/"
# dataset_path = "/media/manuel/New Volume/Computer Vision/fiw"
kin_pairs = ["fd", "ms", "md", "fs"]
kinfacew_set_name = "KinFaceW-II"
kinfacew_n_folds = 5

vgg_path = "/home/manuel/Documents/masters/Computer Vision/kinFaceW/weights/VGG_FACE.t7"

trainer = KinshipTrainer(model_name=model_name, optimizer_name=optimizer_name, lr=lr, momentum=momentum,
                         weight_decay=weight_decay,  n_epochs=n_epochs, dataset=dataset, dataset_path=dataset_path,
                         kin_pairs=kin_pairs, batch_size=batch_size, exp_sufix=exp_sufix, gpu_id=gpu_id,
                         kinfacew_set_name=kinfacew_set_name, kinfacew_n_folds=kinfacew_n_folds,
                         target_metric=target_metric, vgg_weights=vgg_path)
trainer.train()
