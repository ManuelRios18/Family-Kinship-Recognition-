import os
import copy
import tqdm
import torch
import shutil
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from evaluator.evaluator import KinshipEvaluator
from models.small_face_model import SmallFaceModel
from datasets.kinfacew_loader_gen import KinFaceWLoaderGenerator


class KinshipTrainer:

    def __init__(self, model_name, optimizer_name, lr, momentum, weight_decay, n_epochs, dataset, dataset_path,
                 kin_pairs, batch_size, exp_sufix, gpu_id, kinfacew_set_name="KinFaceW-I", kinfacew_n_folds=5,
                 target_metric="acc"):
        self.set_random_seed(990411)
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.kin_pairs = kin_pairs
        self.batch_size = batch_size
        self.exp_sufix = exp_sufix
        self.gpu_id = gpu_id
        self.kinfacew_set_name = kinfacew_set_name
        self.kinfacew_n_folds = kinfacew_n_folds
        self.target_metric = target_metric
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        print("Using device", self.device)
        assert self.dataset in ["kinfacew", "fiw"], f"dataset must be kinfacew or fiw"
        self.transformer_train, self.transformer_test = self.get_transformers()
        self.logs_dir = self.create_log_dir()

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def create_log_dir(self):
        logs_dir = os.path.join("logs", self.exp_sufix + f"_{self.dataset}")
        if os.path.isdir(logs_dir):
            shutil.rmtree(logs_dir)
        os.mkdir(logs_dir)

        return logs_dir

    def get_transformers(self):
        transformer_train = transforms.Compose([transforms.ToPILImage(),
                                                transforms.RandomGrayscale(0.3),
                                                transforms.RandomRotation([-8, +8]),
                                                transforms.ToTensor(),
                                                transforms.RandomHorizontalFlip()])
        transformer_test = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

        return transformer_train, transformer_test

    def train_epoch(self, model, optimizer, criterion, epoch, train_loader, evaluator):
        model.train()
        evaluator.reset()
        for sample in tqdm.tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch}"):
            parent_image, children_image = sample["parent_image"].to(self.device), \
                                           sample["children_image"].to(self.device)
            labels = sample["kin"].to(self.device).float()
            optimizer.zero_grad()
            output = model(parent_image.float(), children_image.float()).squeeze(1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            output = torch.sigmoid(output)
            evaluator.add_batch(list(output.detach().cpu().numpy()), list(labels.detach().cpu().numpy()))
        metrics = evaluator.get_metrics()

    def val_epoch(self, model, epoch, val_loader, evaluator):
        evaluator.reset()
        for sample in tqdm.tqdm(val_loader, total=len(val_loader), desc=f"Val epoch {epoch}"):
            parent_image, children_image = sample["parent_image"].to(self.device), \
                                           sample["children_image"].to(self.device)
            labels = sample["kin"].to(self.device).float()
            output = model(parent_image.float(), children_image.float()).squeeze(1)
            output = torch.sigmoid(output)
            evaluator.add_batch(list(output.detach().cpu().numpy()), list(labels.detach().cpu().numpy()))
        metrics = evaluator.get_metrics()
        return metrics[self.target_metric]

    def train_kinfacew(self):
        kin_loader_gen = KinFaceWLoaderGenerator(dataset_name=self.kinfacew_set_name,
                                                 dataset_path=self.dataset_path)
        for pair_type in self.kin_pairs:
            for fold in range(1, self.kinfacew_n_folds+1):
                test_loader, train_loader = kin_loader_gen.get_data_loader(fold=fold, batch_size=self.batch_size,
                                                                           pair_type=pair_type,
                                                                           train_transformer=self.transformer_train,
                                                                           test_transformer=self.transformer_test)
                print(f"STARTING training for pair {pair_type} fold {fold}")
                best_score = -1
                model = self.load_model()
                model.to(self.device)
                optimizer = self.load_optimizer(model)
                criterion = self.load_criterion()
                train_evaluator = KinshipEvaluator(set_name="Training", pair=pair_type,
                                                   log_path=self.logs_dir, fold=fold)
                test_evaluator = KinshipEvaluator(set_name="Testing", pair=pair_type,
                                                  log_path=self.logs_dir, fold=fold)
                for epoch in range(1, self.n_epochs+1):
                    self.train_epoch(model, optimizer, criterion, epoch, train_loader, train_evaluator)
                    model_score = self.val_epoch(model, epoch, test_loader, test_evaluator)
                    if model_score > best_score:
                        best_score = model_score
                        print(f"NEW best {self.target_metric} score {best_score} for pair {pair_type} in fold {fold}")
                    train_evaluator.save_hist()
                    test_evaluator.save_hist()
                print(f"FINISHING training for pair {pair_type} fold {fold} "
                      f"best {self.target_metric} score {best_score}")

    def train(self):
        if self.dataset == "kinfacew":
            self.train_kinfacew()
        else:
            print("No implemented yet")

    def load_model(self):

        if self.model_name == "small_face_model":
            model = SmallFaceModel()
        else:
            raise Exception("Unkown model")

        return model

    def load_optimizer(self, model):

        if self.optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum,
                                  weight_decay=self.weight_decay)
        else:
            raise Exception("Unkown optimizer")

        return optimizer

    def load_criterion(self):

        criterion = nn.BCEWithLogitsLoss()
        return criterion


