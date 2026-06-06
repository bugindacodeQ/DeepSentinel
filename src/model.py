import torch
import torch.nn as nn
import timm


class DeepfakeDetector(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b4", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.backbone.num_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


def load_model(weights_path: str, model_name: str = "efficientnet_b4", device: str = "cpu") -> DeepfakeDetector:
    model = DeepfakeDetector(model_name=model_name, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model
