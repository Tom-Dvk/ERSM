import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------
# Energy-based Redundancy & Saliency Mask Layer
# ---------------------------------------------------------------
class EnergyMaskLayer(nn.Module):
    """Learnable patch-level energy mask applied to CNN feature maps.

    Each spatial patch is assigned a scalar keep-probability derived from a
    unary energy term (learned linear projection) and a pairwise smoothness
    penalty over the 8-connected neighbourhood.
    """

    def __init__(self, in_channels, patch_size, lambda_u, gamma_p, temperature=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = in_channels * patch_size * patch_size
        self.lambda_u = lambda_u
        self.gamma_p = gamma_p
        self.temperature = temperature
        self.mask_net = nn.Linear(self.input_dim, 1, bias=False)
        self.b_i = None

    def _ensure_bias(self, num_patches, device):
        if self.b_i is None or self.b_i.shape[1] != num_patches:
            self.b_i = nn.Parameter(torch.ones(1, num_patches, device=device) * 6.0)

    def extract_patches(self, x):
        x_unf = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        return x_unf.transpose(1, 2)

    def fold_patches(self, x_unf, output_size):
        x_unf = x_unf.transpose(1, 2)
        return F.fold(x_unf, output_size=output_size,
                      kernel_size=self.patch_size, stride=self.patch_size)

    def _pairwise_feature_penalty_8n(self, p_hat, grid_shape):
        B, N, D = p_hat.shape
        H, W = grid_shape
        p = p_hat.view(B, H, W, D)
        sim_sum = 0.0
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            y0, y1 = max(0, -dy), H - max(0, dy)
            x0, x1 = max(0, -dx), W - max(0, dx)
            yn0, yn1 = y0 + dy, y1 + dy
            xn0, xn1 = x0 + dx, x1 + dx
            src = p[:, y0:y1, x0:x1, :]
            nbr = p[:, yn0:yn1, xn0:xn1, :]
            sim = (src * nbr).sum(-1)
            out = torch.zeros((B, H, W), device=p.device)
            out[:, y0:y1, x0:x1] = sim
            sim_sum += out
        return F.softplus(sim_sum).flatten(1)

    def forward(self, x, forced_mask=None):
        feat_H, feat_W = x.shape[2], x.shape[3]

        if feat_H % self.patch_size != 0 or feat_W % self.patch_size != 0:
            raise ValueError(
                f"Feature map size {feat_H}x{feat_W} not divisible by "
                f"patch_size {self.patch_size}"
            )

        patches = self.extract_patches(x)
        H_grid = feat_H // self.patch_size
        W_grid = feat_W // self.patch_size
        N = patches.shape[1]
        self._ensure_bias(N, x.device)

        eps = 1e-6
        p_hat = patches / (patches.norm(p=2, dim=-1, keepdim=True) + eps)
        z = self.mask_net(p_hat).squeeze(-1) + self.b_i
        keep_prob = torch.sigmoid(-z / max(self.temperature, 1e-6))
        keep_mask = forced_mask if forced_mask is not None else keep_prob

        E_unary = self.lambda_u * F.softplus(z)
        E_pair = (
            self.gamma_p * self._pairwise_feature_penalty_8n(p_hat, (H_grid, W_grid))
            if self.gamma_p > 0
            else torch.zeros_like(E_unary)
        )
        energy = E_unary + E_pair

        patches_masked = patches * keep_mask.unsqueeze(-1)
        x_out = self.fold_patches(patches_masked, (feat_H, feat_W))

        return x_out, {
            "keep_mask": keep_mask,
            "keep_prob": keep_prob,
            "z": z,
            "energy": energy,
            "num_patches": N,
            "grid_shape": (H_grid, W_grid),
        }


# ---------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------
def build_backbone(backbone_name: str):
    """Return ``(backbone, feat_dim)`` for a torchvision model.

    Supported families: ResNet-18/50, ConvNeXt, EfficientNet / EfficientNetV2.
    """
    name = backbone_name.lower()

    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
        )
        return backbone, 2048

    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
        )
        return backbone, 512

    if name in ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]:
        m = getattr(models, name)(weights="DEFAULT")
        return m.features, m.classifier[2].in_features

    if name in [
        "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
        "efficientnet_b6", "efficientnet_b7",
    ]:
        m = getattr(models, name)(weights="DEFAULT")
        return m.features, m.classifier[1].in_features

    raise ValueError(f"Unknown backbone: {backbone_name}")


# ---------------------------------------------------------------
# Frozen-backbone wrapper with optional ERSM
# ---------------------------------------------------------------
class FrozenBackboneWrapper(nn.Module):
    """ImageNet-pretrained backbone (frozen) with a trainable classifier head.

    When ``use_energy=True`` an :class:`EnergyMaskLayer` is inserted between
    the backbone feature map and the global-average-pool + FC head.
    """

    def __init__(self, backbone_name, num_classes, use_energy=False, **energy_kwargs):
        super().__init__()
        self.use_energy = use_energy
        self.backbone_name = backbone_name

        print(f"Initializing {backbone_name} (Frozen Backbone)...")
        self.backbone, self.feat_dim = build_backbone(backbone_name)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.energy_layer = None
        if use_energy:
            self.energy_layer = EnergyMaskLayer(in_channels=self.feat_dim, **energy_kwargs)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def forward_features_pre_mask(self, x):
        return self.backbone(x)

    def forward(self, x, forced_mask=None, scale_compensation=True):
        feat = self.forward_features_pre_mask(x)

        aux = None
        if self.energy_layer is not None:
            feat, aux = self.energy_layer(feat, forced_mask=forced_mask)

        x = self.pool(feat)
        x = torch.flatten(x, 1)

        if scale_compensation and forced_mask is not None and aux is not None:
            N = aux["num_patches"]
            kept = forced_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
            x = x * (N / kept)

        logits = self.fc(x)
        return logits, aux


# ---------------------------------------------------------------
# Utility
# ---------------------------------------------------------------
@torch.no_grad()
def get_feature_hw(backbone_name, device, input_size):
    """Probe the spatial dimensions of the feature map for a given input size."""
    tmp = FrozenBackboneWrapper(backbone_name, num_classes=2, use_energy=False).to(device)
    tmp.eval()
    x = torch.zeros(1, 3, input_size, input_size, device=device)
    feat = tmp.forward_features_pre_mask(x)
    Hf, Wf = feat.shape[-2], feat.shape[-1]
    del tmp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return Hf, Wf
