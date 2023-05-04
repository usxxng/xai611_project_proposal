import torch.nn as nn
from functions import ReverseLayerF
import torchvision.models as models

class CNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        # backbone model
        self.feature = models.densenet121(pretrained=True)

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
