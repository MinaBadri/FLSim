from models.cnn import SimpleCNN
from models.ResNet import SmallResNet


def build_model(cfg: dict):
    """
    Returns the correct model based on config.
    Reads cfg['model']['name'] — 'cnn' or 'resnet'.
    """
    name        = cfg["model"].get("name", "resnet").lower()
    num_classes = cfg["model"]["num_classes"]

    if name == "resnet":
        return SmallResNet(num_classes=num_classes)
    elif name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. Choose 'cnn' or 'resnet'.")