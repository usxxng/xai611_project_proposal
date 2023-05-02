import torch.nn as nn
from functions import ReverseLayerF
import torchvision.models as models


# class CNNModel(nn.Module):
#
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         self.feature = nn.Sequential(
#             nn.Conv2d(3, 8, 3),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.Conv2d(8, 8, 3),
#             nn.BatchNorm2d(8),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(8, 16, 3),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, 3),
#             nn.BatchNorm2d(16),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 3),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 3),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(),
#         )
#
#         self.class_classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(128*2*3, 512),
#             nn.ReLU(),
#             nn.Linear(512, 2),
#         )
#
#         self.domain_classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(128*2*3, 512),
#             nn.ReLU(),
#             nn.Linear(512, 2)
#         )
#
#     def forward(self, input_data, alpha):
#         feature = self.feature(input_data)
#         feature = feature.view(feature.shape[0], -1)
#         reverse_feature = ReverseLayerF.apply(feature, alpha)
#         class_output = self.class_classifier(feature)
#         domain_output = self.domain_classifier(reverse_feature)
#
#         return class_output, domain_output


class CNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        # backbone model
        self.feature = models.resnet34(pretrained=True)

        self.class_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

        self.domain_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(feature.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
