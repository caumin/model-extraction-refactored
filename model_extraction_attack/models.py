"""Module for defining and building various neural network models."""

from __future__ import annotations
import torch
from torch import nn
from typing import Optional, Tuple
from torchvision.models import resnet18, resnet34

# --------- Utility ---------
def _infer_flatten_shape(feats: nn.Module, in_ch: int, img_size: int = 28) -> int:
    """Runs a single dummy forward pass to infer the flattened feature dimension of a model's feature extractor.

    Args:
        feats: The feature extraction part of a neural network.
        in_ch: Number of input channels.
        img_size: Input image size (height and width).

    Returns:
        The flattened dimension size.
    """
    feats.eval()
    with torch.no_grad():
        x = torch.zeros(1, in_ch, img_size, img_size)
        y = feats(x)
        return int(y.view(1, -1).shape[1])

# --------- PRADA-style small CNNs (MNIST/GTSRB) ---------
class PRADA_CNN(nn.Module):
    """
    Generic PRADA CNN used for target/substitute models.
    Follows the architecture described in the PRADA paper (EuroS&P'19).
    """
    def __init__(
        self,
        in_channels: int,
        conv_channels: Tuple[int, ...],
        num_classes: int,
        add_fc100: bool = False,
        img_size: int = 28,
        k: int = 3,
        padding: int = 1,
        dropout_rate: float = 0.0 # New parameter for dropout
    ):
        """Initializes the PRADA_CNN model.

        Args:
            in_channels: Number of input channels.
            conv_channels: Tuple of output channels for each convolutional block.
            num_classes: Number of output classes.
            add_fc100: If True, adds an additional 100-unit fully connected layer.
            img_size: Input image size.
            k: Kernel size for convolutional layers.
            padding: Padding for convolutional layers.
            dropout_rate: Dropout rate to apply (0.0 for no dropout).
        """
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in conv_channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=k, padding=padding, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(), # Add dropout
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        flat_dim = _infer_flatten_shape(self.features, in_channels, img_size)

        cls = [nn.Linear(flat_dim, 200), nn.ReLU(inplace=True), nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()] # Add dropout
        if add_fc100:
            cls += [nn.Linear(200, 100), nn.ReLU(inplace=True), nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()] # Add dropout
            cls += [nn.Linear(100, num_classes)]
        else:
            cls += [nn.Linear(200, num_classes)]
        self.classifier = nn.Sequential(*cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ---- Target models from PRADA Table I ----
def prada_mnist_target(in_ch=1, num_classes=10, img_size=28) -> nn.Module:
    """PRADA MNIST target model (conv2-32 -> pool2 -> conv2-64 -> pool2 -> FC-200 -> FC-10)."""
    return PRADA_CNN(in_channels=in_ch, conv_channels=(32, 64),
                     num_classes=num_classes, add_fc100=False, img_size=img_size, dropout_rate=0.0)

def prada_gtsrb_target(in_ch=3, num_classes=43, img_size=32) -> nn.Module:
    """PRADA GTSRB target model (conv2-64 -> pool2 -> conv2-64 -> pool2 -> FC-200 -> FC-100 -> FC-43)."""
    return PRADA_CNN(in_channels=in_ch, conv_channels=(64, 64),
                     num_classes=num_classes, add_fc100=True, img_size=img_size, dropout_rate=0.0)

# ---- Substitute models from PRADA Table VI (1..5 conv blocks) ----
def prada_sub_cnn1(in_ch=1, num_classes=10, img_size=28, dropout_rate: float = 0.0) -> nn.Module:
    """PRADA substitute CNN with 1 convolutional block."""
    return PRADA_CNN(in_channels=in_ch, conv_channels=(32,), num_classes=num_classes, img_size=img_size, dropout_rate=dropout_rate)

def prada_sub_cnn2(in_ch=1, num_classes=10, img_size=28, dropout_rate: float = 0.0) -> nn.Module:
    """PRADA substitute CNN with 2 convolutional blocks."""
    return PRADA_CNN(in_channels=in_ch, conv_channels=(32, 64), num_classes=num_classes, img_size=img_size, dropout_rate=dropout_rate)

def prada_sub_cnn3(in_ch=1, num_classes=10, img_size=28, dropout_rate: float = 0.0) -> nn.Module:
    """PRADA substitute CNN with 3 convolutional blocks."""
    return PRADA_CNN(in_channels=in_ch, conv_channels=(32, 64, 128), num_classes=num_classes, img_size=img_size, dropout_rate=dropout_rate)

def prada_sub_cnn4(in_ch=1, num_classes=10, img_size=28, dropout_rate: float = 0.0) -> nn.Module:
    """PRADA substitute CNN with 4 convolutional blocks."""
    # Use an increased depth; channels chosen to increase nonlinearity while keeping params moderate
    return PRADA_CNN(in_channels=in_ch, conv_channels=(32, 64, 128, 128), num_classes=num_classes, img_size=img_size, dropout_rate=dropout_rate)

def prada_sub_cnn5(in_ch=1, num_classes=10, img_size=28, dropout_rate: float = 0.0) -> nn.Module:
    """PRADA substitute CNN with 5 convolutional blocks."""
    return PRADA_CNN(in_channels=in_ch, conv_channels=(32, 64, 128, 128, 128), num_classes=num_classes, img_size=img_size, dropout_rate=dropout_rate)

# --------- Existing backbones (minimal re-export to keep file self-contained) ---------
def _xavier_init(m):
    """Initializes model weights using Xavier uniform distribution."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Xie2019(nn.Module):
    """Xie et al. (2019) model architecture."""
    def __init__(self, num_classes=1, in_channels=3, dropout_rate=0.6, img_size=224):
        """Initializes the Xie2019 model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_rate: Dropout rate.
            img_size: Input image size.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 11, padding=5, stride=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, padding=1, stride=2), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024), nn.ReLU(True), nn.Dropout(dropout_rate),
            nn.Linear(1024, 512), nn.ReLU(True), nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def _torchvision_resnet(resnet_builder, in_ch: int, num_classes: int, img_size: int) -> nn.Module:
    """Helper function to build a torchvision ResNet model with custom input channels and output classes."""
    # pretrained=False for fair comparison from scratch
    model = resnet_builder(weights=None, num_classes=1000)
    if in_ch != 3:
        # Adapt first conv layer for non-RGB images
        model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Adapt final FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Factory & registry
_ARCHS = {
    # PRADA targets
    "prada_mnist_target": lambda in_ch, num_classes, img: prada_mnist_target(in_ch, num_classes, img),
    "prada_gtsrb_target": lambda in_ch, num_classes, img: prada_gtsrb_target(in_ch, num_classes, img),
    # PRADA substitutes (Table VI)
    "prada_sub_cnn1": lambda in_ch, num_classes, img, dropout_rate: prada_sub_cnn1(in_ch, num_classes, img, dropout_rate),
    "prada_sub_cnn2": lambda in_ch, num_classes, img, dropout_rate: prada_sub_cnn2(in_ch, num_classes, img, dropout_rate),
    "prada_sub_cnn3": lambda in_ch, num_classes, img, dropout_rate: prada_sub_cnn3(in_ch, num_classes, img, dropout_rate),
    "prada_sub_cnn4": lambda in_ch, num_classes, img, dropout_rate: prada_sub_cnn4(in_ch, num_classes, img, dropout_rate),
    "prada_sub_cnn5": lambda in_ch, num_classes, img, dropout_rate: prada_sub_cnn5(in_ch, num_classes, img, dropout_rate),
    # Existing example
    "resnet18": lambda in_ch, num_classes, img: _torchvision_resnet(resnet18, in_ch, num_classes, img),
    "resnet34": lambda in_ch, num_classes, img: _torchvision_resnet(resnet34, in_ch, num_classes, img),
    "xie2019": lambda in_ch, num_classes, img: Xie2019(num_classes=num_classes, in_channels=in_ch, img_size=img),
}

def build_model(arch: str, num_classes: int, in_channels: int, device="cpu", img_size: Optional[int]=None, dropout_rate: float = 0.0) -> nn.Module:
    """Builds a model from the registry based on the architecture name.

    Args:
        arch: Architecture name (e.g., 'resnet18', 'prada_mnist_target').
        num_classes: Number of output classes.
        in_channels: Number of input channels.
        device: The device to load the model onto.
        img_size: Input image size (optional, used for some models).
        dropout_rate: Dropout rate for models that support it.

    Returns:
        An instance of the specified model.

    Raises:
        ValueError: If an unknown architecture name is provided.
    """
    print(f"[DEBUG] build_model called with arch={arch}, num_classes={num_classes}, in_channels={in_channels}, img_size={img_size}, dropout_rate={dropout_rate}")
    if img_size is None:
        img_size = 28 if in_channels == 1 else 224
    if arch not in _ARCHS:
        raise ValueError(f"Unknown arch: {arch}. Available: {list(_ARCHS.keys())}")
    
    # Pass dropout_rate if the model constructor accepts it
    if arch.startswith("prada_sub_cnn"):
        model = _ARCHS[arch](in_channels, num_classes, img_size, dropout_rate=dropout_rate)
    else:
        model = _ARCHS[arch](in_channels, num_classes, img_size)
    return model.to(device)

def load_checkpoint(model: nn.Module, path: str, map_location="cpu", strict: bool=False) -> dict:
    """Loads a model checkpoint from the specified path.

    Args:
        model: The model to load the state_dict into.
        path: Path to the checkpoint file.
        map_location: Specifies how to remap storage locations.
        strict: Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() method.

    Returns:
        The loaded state_dict.
    """
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        # Strip common prefixes (e.g., 'module.')
        state = {k.replace("module.", ""): v for k, v in state.items()}
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    return state