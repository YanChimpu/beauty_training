from torchvision.models import resnet50
import torch.nn as nn
from classy_vision.models import ClassyModel, register_model


@register_model("my_model")
class MyModel(ClassyModel):
    def __init__(self):
        super().__init__()
        self.resnet = ClassyModel.from_model(resnet50())
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1000, 8)

    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(x)
        x = self.linear(x)
        return x

    @classmethod
    def from_config(cls, config):
        return cls()

