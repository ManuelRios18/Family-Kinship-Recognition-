import torch
import operator
import functools
import torch.nn as nn
import torch.nn.functional as f


class SmallSiameseFaceModel(nn.Module):

    def __init__(self):
        super(SmallSiameseFaceModel, self).__init__()

        self.conv_1_1 = nn.Conv2d(3, 16, kernel_size=5)
        self.bn_1_1 = nn.BatchNorm2d(16)
        # nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=2, stride=2),
        self.conv_2_1 = nn.Conv2d(16, 64, kernel_size=5)
        self.bn_2_1 = nn.BatchNorm2d(64)
        # nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=2, stride=2),
        self.conv_2_2 = nn.Conv2d(64, 128, kernel_size=5)
        self.bn_2_2 = nn.BatchNorm2d(128)

        self.classifier = nn.Sequential(
            nn.Linear(10368, 640),
            nn.ReLU(),
            nn.Linear(640, 1)
        )

    def forward(self, parent_image, children_image):

        parent_features = self.features(parent_image.float())
        children_features = self.features(children_image.float())
        distance = torch.abs(parent_features - children_features)
        x = torch.flatten(distance, 1)
        x = f.normalize(x, dim=0, p=2)
        x = self.classifier(x)

        return x

    def encode_face(self, input_face):
        out_conv_1_1 = self.bn_1_1(self.conv_1_1(input_face))
        out_conv_1_1 = f.max_pool2d(f.relu(out_conv_1_1), 2, 2)

        out_conv_2_1 = self.bn_2_1(self.conv_2_1(out_conv_1_1))
        out_conv_2_1 = f.max_pool2d(f.relu(out_conv_2_1), 2, 2)

        out_conv_2_2 = self.bn_2_2(self.conv_2_2(out_conv_2_1))

        return out_conv_1_1, out_conv_2_2

    def get_fetures(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        large_features = self.large_scale_features(x)
        large_features = torch.flatten(large_features, 1)
        features = torch.cat((features, large_features), 1)
        return features

