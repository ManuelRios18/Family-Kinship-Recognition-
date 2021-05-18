import torch
import operator
import functools
import torch.nn as nn


class SmallSiameseFaceModel(nn.Module):

    def __init__(self):
        super(SmallSiameseFaceModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128)
        )
        n_features = functools.reduce(operator.mul, list(self.features(torch.rand(1, *(3, 64, 64))).shape))
        print("N features", n_features)
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 640),
            nn.ReLU(),
            nn.Linear(640, 1)
        )
        self.large_scale_features = nn.Sequential(*list(self.features.children())[:-4])

    def forward(self, parent_image, children_image):

        parent_features = self.features(parent_image.float())
        children_features = self.features(children_image.float())
        distance = torch.abs(parent_features - children_features)
        x = torch.flatten(distance, 1)
        x = self.classifier(x)

        return x

    def get_fetures(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        large_features = self.large_scale_features(x)
        large_features = torch.flatten(large_features, 1)
        features = torch.cat((features, large_features), 1)
        return features
