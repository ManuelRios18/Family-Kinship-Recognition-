import os
import copy
import tqdm
import torch
import shutil
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torchvision import transforms
from models.kin_facenet import KinFaceNet
from datasets.fiw_dataset import FIWDataset
from evaluator.evaluator import KinshipEvaluator
from models.small_face_model import SmallFaceModel
from models.vgg_face_siamese import VGGFaceSiamese
from models.vgg_face_multichannel import VGGFaceMutiChannel
from datasets.kinfacew_loader_gen import KinFaceWLoaderGenerator
from models.small_siamese_face_model import SmallSiameseFaceModel


class KinshipTrainer:

    def __init__(self, model_name, optimizer_name, lr, momentum, weight_decay, n_epochs, dataset, dataset_path,
                 kin_pairs, batch_size, exp_sufix, gpu_id, kinfacew_set_name="KinFaceW-I", kinfacew_n_folds=5,
                 target_metric="acc", vgg_weights=None):
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
        self.vgg_weights = vgg_weights
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
        if self.dataset == "kinfacew":
            sufix = "_II"
            if self.kinfacew_set_name == "KinFaceW-I":
                sufix = "_I"
            logs_dir += sufix
        if os.path.isdir(logs_dir):
            shutil.rmtree(logs_dir)
        os.mkdir(logs_dir)

        return logs_dir

    def get_transformers(self):
        if "vgg" in self.model_name:
            transformer_train = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize((224, 224)),
                                                    transforms.RandomGrayscale(0.3),
                                                    transforms.RandomRotation([-8, +8]),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        [129.1863 / 255, 104.7624 / 255, 93.5940 / 255],
                                                        [1 / 255, 1 / 255, 1 / 255]),
                                                    transforms.RandomHorizontalFlip()])
            transformer_test = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([129.1863 / 255, 104.7624 / 255, 93.5940 / 255],
                                                                        [1 / 255, 1 / 255, 1 / 255])])
        elif "facenet" in self.model_name:

            transformer_train = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize((160, 160)),
                                                    transforms.RandomGrayscale(0.3),
                                                    transforms.RandomRotation([-8, +8]),
                                                    transforms.ToTensor(),
                                                    transforms.RandomHorizontalFlip()])

            transformer_test = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize((160, 160)),
                                                   transforms.ToTensor()])

        else:
            transformer_train = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize((64, 64)),
                                                    transforms.RandomGrayscale(0.3),
                                                    transforms.RandomRotation([-8, +8]),
                                                    transforms.ToTensor(),
                                                    transforms.RandomHorizontalFlip()])
            transformer_test = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize((64, 64)),
                                                   transforms.ToTensor()])

        return transformer_train, transformer_test,

    def custom_loss(self, emb_a, emb_b, y):
        alpha = 2.5
        emb_a = f.normalize(emb_a)
        emb_b = f.normalize(emb_b)
        ind = list()
        for a_i, anchor in enumerate(emb_a):
            anchor = anchor.repeat(emb_a.size()[0], 1)
            neg_distances = f.pairwise_distance(anchor, emb_b, 2)
            neg_distances[a_i] = float("Inf")
            hardest_idx = torch.argmin(neg_distances).item()
            ind.append(hardest_idx)
        emb_n = emb_b[[ind]]
        dist = f.pairwise_distance(emb_a, emb_b, 2) - f.pairwise_distance(emb_a, emb_n, 2) + 1
        dist = f.relu(y*dist)
        
        return torch.mean(dist)

    def train_epoch(self, model, optimizer, criterion, epoch, train_loader, evaluator):
        model.train()
        evaluator.reset()
        for sample in tqdm.tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch}"):
            parent_image, children_image = sample["parent_image"].to(self.device), \
                                           sample["children_image"].to(self.device)
            labels = sample["kin"].to(self.device).float()
            optimizer.zero_grad()
            output, p_f, c_f = model(parent_image.float(), children_image.float())
            output, p_f, c_f = output.squeeze(1), p_f.squeeze(1), c_f.squeeze(1)
            loss = criterion(output, labels) + self.custom_loss(p_f, c_f, labels)
            loss.backward()
            optimizer.step()
            output = torch.sigmoid(output)
            evaluator.add_batch(list(output.detach().cpu().numpy()), list(labels.detach().cpu().numpy()))
        metrics = evaluator.get_metrics(self.target_metric)

    def val_epoch(self, model, epoch, val_loader, evaluator):
        evaluator.reset()
        model.eval()
        for sample in tqdm.tqdm(val_loader, total=len(val_loader), desc=f"Val epoch {epoch}"):
            parent_image, children_image = sample["parent_image"].to(self.device), \
                                           sample["children_image"].to(self.device)
            labels = sample["kin"].to(self.device).float()
            output, _, _ = model(parent_image.float(), children_image.float())
            output = output.squeeze(1)
            output = torch.sigmoid(output)
            evaluator.add_batch(list(output.detach().cpu().numpy()), list(labels.detach().cpu().numpy()))
        metrics = evaluator.get_metrics(self.target_metric)
        return metrics[self.target_metric]

    def train_kinfacew(self):
        kin_loader_gen = KinFaceWLoaderGenerator(dataset_name=self.kinfacew_set_name,
                                                 dataset_path=self.dataset_path,
                                                 color_space_name=self.get_color_space_name())
        for pair_type in self.kin_pairs:
            pair_evaluators = list()
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
                        test_evaluator.save_best_metrics()
                        self.save_model(model, f"best_model_{pair_type}_fold_{fold}")
                        print(f"NEW best {self.target_metric} score {best_score} for pair {pair_type} in fold {fold}")
                    train_evaluator.save_hist()
                    test_evaluator.save_hist()
                print(f"FINISHING training for pair {pair_type} fold {fold} "
                      f"best {self.target_metric} score {best_score}")
                pair_evaluators.append(test_evaluator)
            pair_evaluator = KinshipEvaluator(set_name="Testing", pair=pair_type, log_path=self.logs_dir)
            pair_evaluator.get_kinface_pair_metrics(pair_evaluators, pair_type)

    def fiw_triplet_loss(self, parents_embedings, children_embedings, y, parents_families, children_families):
        # Seteamos margin
        margin = 2
        # Copiamos y por que me da miedito cagarmela y perder el gradiente
        y_copy = y.cpu().numpy()
        # Lista vacia donde almacenamos los indices de las parejas dificies par cada padre
        hard_negative_index = list()
        for parent_index, parent_emb in enumerate(parents_embedings):
            # Sacamos la familia del padre
            parent_family = parents_families[parent_index]
            # Hacemos una mascara donde 1 significa que el padre y el hijo son de la misma familia
            child_families_mask = np.array([1 if parent_family == child_family else 0
                                            for child_family in children_families])
            # Repetimos el vector del papa tantas veces como sea necesario para operar
            anchor = parent_emb.repeat(children_embedings.size()[0], 1)
            # Sacamos todas las distancias
            distances = f.pairwise_distance(anchor, children_embedings, 2).detach().cpu().numpy()
            # Volemos Infinito las distancias donde la mascara sea 1 ( es decir las de la misma familia)
            # Por lo tanto no vamos a sacar indices de parejas dificiles pertenicientes a la misma familia
            distances[child_families_mask == 1] = float("Inf")
            # Guardamos el Indice del hijo dificil
            hard_negative_index.append(np.argmin(distances))
        # Volvemos los indices un array
        hard_negative_index = np.array(hard_negative_index)

        # Como solo nos interesan como anclas las parejs que tienen la etiqueta positiva, filtramos con y
        anchors = parents_embedings[y_copy == 1]
        positive = children_embedings[y_copy == 1]
        # Antes de filtrar las negativas primero ordenamos con los indices dificiles
        negative = children_embedings[[hard_negative_index]][y_copy == 1]
        # Hacemos la formulita
        triplet_loss = f.relu(f.pairwise_distance(anchors, positive, 2) -
                              f.pairwise_distance(anchors, negative, 2) + margin)
        # retornamos la media
        return torch.mean(triplet_loss)

    def train_epoch_fiw(self, model, optimizer, criterion, epoch, train_loader, evaluator):
        model.train()
        evaluator.reset()
        for sample in tqdm.tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch}"):
            parent_image, children_image = sample["parent_image"].to(self.device), \
                                           sample["children_image"].to(self.device)
            labels = sample["kin"].to(self.device).float()
            optimizer.zero_grad()
            output, parents_features, children_features = model(parent_image.float(), children_image.float())
            output, parents_features, children_features = output.squeeze(1), parents_features.squeeze(1), \
                                                          children_features.squeeze(1)
            triplet_loss = self.fiw_triplet_loss(parents_embedings=parents_features,
                                                 children_embedings=children_features,
                                                 y=labels, parents_families=sample["parent_family_id"],
                                                 children_families=sample["children_family_id"])
            loss = criterion(output, labels) + triplet_loss
            loss.backward()
            optimizer.step()
            output = torch.sigmoid(output)
            evaluator.add_batch(list(output.detach().cpu().numpy()), list(labels.detach().cpu().numpy()))
        metrics = evaluator.get_metrics(self.target_metric)

    def val_epoch_fiw(self, model, epoch, val_loader, evaluator):
        evaluator.reset()
        model.eval()
        for sample in tqdm.tqdm(val_loader, total=len(val_loader), desc=f"Val epoch {epoch}"):
            parent_image, children_image = sample["parent_image"].to(self.device), \
                                           sample["children_image"].to(self.device)
            labels = sample["kin"].to(self.device).float()
            output, _, _ = model(parent_image.float(), children_image.float())
            output = output.squeeze(1)
            output = torch.sigmoid(output)
            evaluator.add_batch(list(output.detach().cpu().numpy()), list(labels.detach().cpu().numpy()))
        metrics = evaluator.get_metrics(self.target_metric)
        return metrics[self.target_metric]

    def train_fiw(self):
        for pair_type in self.kin_pairs:
            train_loader = torch.utils.data.DataLoader(FIWDataset(self.dataset_path, pair_type,
                                                                  "train",
                                                                  self.transformer_train,
                                                                  color_space=self.get_color_space_name()),
                                                       batch_size=self.batch_size,
                                                       shuffle=True, num_workers=4)
            test_loader = torch.utils.data.DataLoader(FIWDataset(self.dataset_path, pair_type,
                                                                 "val", self.transformer_test,
                                                                 color_space=self.get_color_space_name()),
                                                      batch_size=self.batch_size,
                                                      shuffle=True, num_workers=4)
            print(f"STARTING training for pair {pair_type}")
            best_score = -1
            model = self.load_model()
            model.to(self.device)
            optimizer = self.load_optimizer(model)
            criterion = self.load_criterion()
            train_evaluator = KinshipEvaluator(set_name="Training", pair=pair_type, log_path=self.logs_dir)
            test_evaluator = KinshipEvaluator(set_name="Testing", pair=pair_type, log_path=self.logs_dir)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,8)
            for epoch in range(1, self.n_epochs + 1):
                self.train_epoch_fiw(model, optimizer, criterion, epoch, train_loader, train_evaluator)
                model_score = self.val_epoch_fiw(model, epoch, test_loader, test_evaluator)
                if model_score > best_score:
                    best_score = model_score
                    test_evaluator.save_best_metrics()
                    self.save_model(model, f"best_model_{pair_type}")
                    print(f"NEW best {self.target_metric} score {best_score} for pair {pair_type}")
                scheduler.step()
                train_evaluator.save_hist()
                test_evaluator.save_hist()
            print(f"FINISHING training for pair {pair_type}"
                  f"best {self.target_metric} score {best_score}")

    def save_model(self, model, model_name):
        with open(f"{self.logs_dir}/{model_name}.pth", 'wb') as fp:
            state = model.state_dict()
            torch.save(state, fp)

    def train(self):
        if self.dataset == "kinfacew":
            self.train_kinfacew()
        else:
            self.train_fiw()

    def load_model(self):

        if self.model_name == "small_face_model":
            model = SmallFaceModel()
        elif self.model_name == "small_siamese_face_model":
            model = SmallSiameseFaceModel()
        elif self.model_name == "vgg_multichannel":
            model = VGGFaceMutiChannel(self.vgg_weights)
        elif self.model_name == "vgg_siamese":
            model = VGGFaceSiamese(self.vgg_weights)
        elif self.model_name == "kin_facenet":
            model = KinFaceNet()
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

    def get_color_space_name(self):
        color_space = "rgb"
        if "vgg" in self.model_name:
            color_space = "bgr"
        print("Setting Color space to:", color_space)
        return color_space
