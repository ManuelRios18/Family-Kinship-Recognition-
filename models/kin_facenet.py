import torch
import torch.nn as nn
import torch.nn.functional as f
from facenet_pytorch import InceptionResnetV1


class KinFaceNet(nn.Module):

    def __init__(self):
        super(KinFaceNet, self).__init__()
        self.backbone = InceptionResnetV1(pretrained='vggface2')
        self.classifier = nn.Sequential(
            nn.Linear(512, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, parent_image, children_image):
        parent_features = self.backbone(parent_image)
        children_features = self.backbone(children_image)
        distance = torch.abs(parent_features - children_features)
        x = torch.flatten(distance, 1)
        x = self.classifier(x)

        return x, torch.flatten(parent_features, 1), torch.flatten(children_features, 1)
