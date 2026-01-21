import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from omegaconf import DictConfig

###############################################################################
# --------------------  Loss functions  ------------------------------------- #
###############################################################################

class AMSRPlusLoss(nn.Module):
    """Adaptive Multi-Component Soft Regularisation (AMSR+) loss."""
    def __init__(self, base_loss_fn: nn.Module = None, kl_weight: float = 0.5):
        super().__init__()
        self.base_loss_fn = base_loss_fn if base_loss_fn else nn.CrossEntropyLoss()
        self.kl_weight = kl_weight
        self.aux_net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2), nn.Sigmoid()
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aug_params: torch.Tensor):
        ce = self.base_loss_fn(logits, targets)
        batch, num_classes = logits.shape
        one_hot = F.one_hot(targets, num_classes=num_classes).float()

        coeffs = self.aux_net(aug_params)  # B×2 in [0,1]
        comp1, comp2 = coeffs[:, 0], coeffs[:, 1]
        base_exp = 1.0 - 0.5 * comp1
        refined_exp = base_exp - 0.3 * comp2
        exp = refined_exp.clamp(0.1, 1.0).unsqueeze(1)  # B×1

        soft_targets = (one_hot ** exp)
        soft_targets = soft_targets / soft_targets.sum(1, keepdim=True)
        kl = F.kl_div(F.log_softmax(logits, dim=1), soft_targets, reduction="batchmean")
        return ce + self.kl_weight * kl


class BaselineLoss(nn.Module):
    """Standard cross-entropy for baseline (fixed softening)."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets)

###############################################################################
# --------------------------  Model builder  --------------------------------- #
###############################################################################

def get_model(cfg: DictConfig) -> nn.Module:
    if cfg.model.name.lower() == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 classes
    else:
        raise ValueError(f"Unsupported model {cfg.model.name}")
    return model
