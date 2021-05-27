import tqdm
import torch
import utils
import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from models.vgg_face_siamese import VGGFaceSiamese
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from datasets.kinfacew_loader_gen import KinFaceWLoaderGenerator


def get_metrics(parameters, cv, X, y, name):
    y = np.array(y)
    y_trues = list()
    y_probas = list()
    folds_names = list()
    for i, (train_index, val_index) in enumerate(cv):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]
        clf_2 = RandomForestClassifier(**parameters)
        clf_2.fit(X_train, y_train)
        y_val_hat_p = clf_2.predict_proba(X_val)[:, 1]
        y_trues += list(y_val)
        y_probas += list(y_val_hat_p)
        folds_names += [i + 1] * X_val.shape[0]
        y_hat = np.array([0] * X_val.shape[0])
        y_hat[y_val_hat_p > 0.5] = 1
        print("f1-score", name, f1_score(y_val, y_hat))
        clf_2 = None
    result = pd.DataFrame({"target": y_trues, "prob": y_probas, "fold": folds_names})
    result.to_csv(f"results/{name}.csv", index=False)


def get_descriptors(model, loader, device=None):

    all_features = list()
    all_labels = list()
    all_folds = list()
    for batch_idx, sample in tqdm.tqdm(enumerate(loader), total=len(loader), ncols=80, leave=False):
        parent_images = sample["parent_image"]
        children_images = sample["children_image"]
        if device:
            parent_images = parent_images.to(device)
            children_images = children_images.to(device)
        features = model.get_features(parent_images, children_images)
        all_features.append(features)
        labels = sample["kin"]
        folds = sample["fold"].tolist()
        all_labels += labels
        all_folds += folds

    all_features = np.vstack(all_features)
    return all_features, all_labels, all_folds


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device:", device)
print("Loading Model")
model = VGGFaceSiamese()
### .double() ??
model.to(device)

weights_path = "/home/manuel/Documents/masters/Computer Vision/kinFaceW/weights/VGG_FACE.t7"
model.load_weights(weights_path)
model.eval()
print("Model Loaded")

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

kin_df = "KinFaceW-II"
dataset_path = "/home/manuel/Documents/masters/Computer Vision/YGYME/data/"
kin_data = KinFaceWLoaderGenerator(kin_df, dataset_path, "bgr")
for pair_type in ["fs", "fd", "ms", "md"]:
    kin_face_w_loader = kin_data.get_data_loader_full(batch_size=1,
                                                      pair_type=pair_type,
                                                      transformer=transformer_test)

    X, y, folds = get_descriptors(model, kin_face_w_loader, device)

    clf = RandomForestClassifier()
    param_grid = {
            "max_depth": [20, 30,  40, 45, 50, 60],
            "n_estimators": [250, 450, 500, 550, 600, 650],
            "random_state": [1118]
        }
    cv = [
        (np.array([i for i, v in enumerate(folds) if v != 1]), np.array([i for i, v in enumerate(folds) if v == 1])),
        (np.array([i for i, v in enumerate(folds) if v != 2]), np.array([i for i, v in enumerate(folds) if v == 2])),
        (np.array([i for i, v in enumerate(folds) if v != 3]), np.array([i for i, v in enumerate(folds) if v == 3])),
        (np.array([i for i, v in enumerate(folds) if v != 4]), np.array([i for i, v in enumerate(folds) if v == 4])),
        (np.array([i for i, v in enumerate(folds) if v != 5]), np.array([i for i, v in enumerate(folds) if v == 5]))
    ]
    print("Starting grid search for pair type", pair_type)

    scoring = {'F1-Score': 'f1_micro', 'Accuracy': make_scorer(accuracy_score)}
    grid_search = GridSearchCV(estimator=clf, cv=cv, param_grid=param_grid, scoring=scoring, refit='F1-Score')
    grid_search.fit(X, y)
    print("Best score after grid search", grid_search.best_score_)
    print("Best parameters after grid search", grid_search.best_params_)
    get_metrics(grid_search.best_params_, cv, X, y, f"{kin_df}_{pair_type}")
print("---"*100)
