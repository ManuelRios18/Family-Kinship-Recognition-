import os
import utils
import torch
import tqdm
import random
import matplotlib
import numpy as np
from torchvision import transforms
from evaluator.evaluator import KinshipEvaluator
from datasets.kinfacew_loader_gen import KinFaceWLoaderGenerator
from models.small_siamese_face_model import SmallSiameseFaceModel
from sklearn.metrics import accuracy_score
matplotlib.use('TkAgg')

seed = 990411
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

dataset_name = "KinFaceW-I"
dataset_path = "/home/manuel/Documents/masters/Computer Vision/YGYME/data/"

model_path = f"logs/triplet_bn_auc/"
# model_path = f"logs/no_triplet_bn_auc/"
loader_gen = KinFaceWLoaderGenerator(dataset_name=dataset_name, dataset_path=dataset_path, color_space_name="rgb")

pair_type = "fd"
fold = 1
gpu_id = 0

# ms fold 3

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
# device = "cpu"
distances = list()

model = SmallSiameseFaceModel()
model.eval()
state_dict = torch.load(os.path.join(model_path, f"best_model_{pair_type}_fold_{fold}.pth"))
model.load_state_dict(state_dict)
model.to(device)
transformer = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Resize((64, 64)),
                                  transforms.ToTensor()])


for fold in [1, 2, 3, 4, 5]:
    model = SmallSiameseFaceModel()
    model.eval()
    state_dict = torch.load(os.path.join(model_path, f"best_model_{pair_type}_fold_{fold}.pth"))
    model.load_state_dict(state_dict)
    model.to(device)

    loader_test, loader_train = loader_gen.get_data_loader(fold, 16, pair_type, transformer, transformer)

    evaluator = KinshipEvaluator(set_name="Training", pair=pair_type, log_path="logs/testing", fold=fold)
    evaluator.reset()
    parent_features = list()
    child_features = list()
    model_out = list()
    targets = list()

    for sample in tqdm.tqdm(loader_test, total=len(loader_test), desc=f"Val epoch {1}"):
        parent_image, children_image = sample["parent_image"].to(device), \
                                       sample["children_image"].to(device)
        labels = sample["kin"].to(device).float()
        output, p_f, c_f = model(parent_image.float(), children_image.float())
        output, p_f, c_f = output.squeeze(1), p_f.squeeze(1).detach().cpu().numpy(), c_f.squeeze(1).detach().cpu().numpy()
        parent_features.append(p_f)
        child_features.append(c_f)
        output = torch.sigmoid(output)
        model_out += list(output.detach().cpu().numpy())
        targets += list(labels.detach().cpu().numpy())
    best_threshold = utils.load_json(f"{model_path}/{pair_type}_testing_fold_{fold}.json")["best_threshold"]

    targets = np.array(targets)
    y_hat = np.zeros_like(model_out)
    y_hat[np.array(model_out) > 0.5338352918624878] = 1
    acc = accuracy_score(targets, y_hat)
    print("Accuracy ", acc)
