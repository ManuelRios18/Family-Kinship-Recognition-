from trainers.kinship_trainer import KinshipTrainer

exp_sufix = "small_face_model"
n_epochs = 150
lr = 0.001
batch_size = 16
momentum = 0.9
weight_decay = 0.005
gpu_id = 0
optimizer_name = "SGD"
target_metric = "acc"

dataset = "kinfacew"
dataset_path = "/home/manuel/Documents/masters/Computer Vision/YGYME/data/"
kin_pairs = ["fd", "ms", "md", "fs"]
kinfacew_set_name = "KinFaceW-II"
kinfacew_n_folds = 5


model_name = "small_face_model"
trainer = KinshipTrainer(model_name=model_name, optimizer_name=optimizer_name, lr=lr, momentum=momentum,
                         weight_decay=weight_decay,  n_epochs=n_epochs, dataset=dataset, dataset_path=dataset_path,
                         kin_pairs=kin_pairs, batch_size=batch_size, exp_sufix=exp_sufix, gpu_id=gpu_id,
                         kinfacew_set_name=kinfacew_set_name, kinfacew_n_folds=kinfacew_n_folds,
                         target_metric=target_metric)
trainer.train()