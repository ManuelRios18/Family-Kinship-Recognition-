import torch
import torchfile
import torch.nn as nn
import torch.nn.functional as F


class VGGFaceMutiChannel(nn.Module):

    def __init__(self, vgg_weights_path=None):
        super(VGGFaceMutiChannel, self).__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(6, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, 1)
        if vgg_weights_path is not None:
            print("Trying to load VGG weights")
            self.load_weights(vgg_weights_path)

    def forward(self, parent_image, children_image):
        input_tensor = torch.cat((parent_image, children_image), 1)
        x = self.encode_faces(input_tensor.float())
        x = F.relu(self.fc7(x))
        x = self.classifier(x)

        return x

    def encode_faces(self, x):

        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)

        return x

    def load_weights(self, path):
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    try:
                        self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                        counter += 1
                        if counter > self.block_size[block - 1]:
                            counter = 1
                            block += 1
                        self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                        self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                    except:
                        print("Skiping conv_%d_%d" % (block, counter))
                else:
                    try:
                        self_layer = getattr(self, "fc%d" % (block))
                        block += 1
                        self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                        self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                    except:
                        print("Skipping fc%d" % (block))