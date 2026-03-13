import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM implementation with automatic hook management."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=None, forced_mask=None):
        x = x.detach().requires_grad_(True)
        logits, _ = self.model(x, forced_mask=forced_mask)

        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        B = logits.size(0)
        score = logits[torch.arange(B, device=logits.device), class_idx].sum()

        self.model.zero_grad()
        score.backward()

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)

        cam_min = cam.flatten(1).min(dim=1)[0].view(B, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(B, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)

        return cam


def deletion_auc_compare(model_energy, model_base, loader, device,
                         out_path_curve=None):
    """Deletion curve + AUC: ERSM vs GradCAM vs Random.

    All deletion curves are evaluated on ``model_energy``.  GradCAM saliency
    is computed on ``model_base`` (same frozen backbone, but the FC head was
    trained on full un-masked features so its gradient scale is correct).
    """
    print("   -> Running Deletion AUC comparison (ERSM vs GradCAM vs Random)...")

    model_energy.eval()
    model_base.eval()

    gradcam = GradCAM(model_base, model_base.backbone[-1])

    steps = None
    correct_e = correct_g = correct_r = total = None

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        B = imgs.size(0)

        with torch.no_grad():
            _, aux = model_energy(imgs)
            N = aux["z"].shape[1]
            energy_rank = torch.argsort(aux["z"], dim=1, descending=True)

        cam = gradcam.generate(imgs)

        with torch.no_grad():
            cam_patches = model_energy.energy_layer.extract_patches(
                cam.unsqueeze(1)
            ).mean(dim=-1)
            gradcam_rank = torch.argsort(cam_patches, dim=1, descending=False)
            random_rank = torch.stack(
                [torch.randperm(N, device=device) for _ in range(B)]
            )

            if steps is None:
                steps = list(range(0, N + 1, max(1, N // 40)))
                if steps[-1] != N:
                    steps.append(N)
                correct_e = [0] * len(steps)
                correct_g = [0] * len(steps)
                correct_r = [0] * len(steps)
                total = 0

            total += B

            for i, k in enumerate(steps):
                mask_e = torch.ones(B, N, device=device)
                mask_g = torch.ones(B, N, device=device)
                mask_r = torch.ones(B, N, device=device)
                if k > 0:
                    mask_e.scatter_(1, energy_rank[:, :k], 0)
                    mask_g.scatter_(1, gradcam_rank[:, :k], 0)
                    mask_r.scatter_(1, random_rank[:, :k], 0)

                out_e, _ = model_energy(imgs, forced_mask=mask_e,
                                        scale_compensation=False)
                out_g, _ = model_energy(imgs, forced_mask=mask_g,
                                        scale_compensation=False)
                out_r, _ = model_energy(imgs, forced_mask=mask_r,
                                        scale_compensation=False)

                correct_e[i] += (out_e.argmax(1) == labels).sum().item()
                correct_g[i] += (out_g.argmax(1) == labels).sum().item()
                correct_r[i] += (out_r.argmax(1) == labels).sum().item()

    acc_e = [100.0 * c / total for c in correct_e]
    acc_g = [100.0 * c / total for c in correct_g]
    acc_r = [100.0 * c / total for c in correct_r]

    frac_steps = [k / max(steps) for k in steps]

    if out_path_curve is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(frac_steps, acc_e, linestyle="-", color="red",
                 label="ERSM Deletion")
        plt.plot(frac_steps, acc_g, linestyle="--", color="orange",
                 label="GradCAM Deletion")
        plt.plot(frac_steps, acc_r, linestyle=":", color="blue",
                 label="Random Deletion")
        plt.xlabel("Fraction of Patches Deleted")
        plt.ylabel("Accuracy (%)")
        plt.title("Deletion Curve: ERSM vs GradCAM vs Random\n"
                   "(least important deleted first — higher AUC = better)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path_curve)
        plt.close()

    gradcam.remove_hooks()

    auc_e = float(np.trapz(acc_e, frac_steps))
    auc_g = float(np.trapz(acc_g, frac_steps))
    auc_r = float(np.trapz(acc_r, frac_steps))

    return {
        "auc_energy": auc_e,
        "auc_gradcam": auc_g,
        "auc_random": auc_r,
        "deletion_curve": {
            "frac_steps": frac_steps,
            "acc_energy": acc_e,
            "acc_gradcam": acc_g,
            "acc_random": acc_r,
        },
    }


def compare_sparsity_entropy(model_energy, model_base, loader, device):
    """Entropy and sparsity comparison between ERSM and GradCAM.

    GradCAM is run on ``model_base`` (clean backbone, no mask layer) and
    pooled to the same patch granularity as ERSM.
    """
    print("   -> Computing sparsity / entropy metrics...")

    model_energy.eval()
    model_base.eval()

    gradcam = GradCAM(model_base, model_base.backbone[-1])

    entropy_energy, entropy_gradcam = [], []
    active_energy, active_gradcam = [], []

    for imgs, _ in loader:
        imgs = imgs.to(device)

        with torch.no_grad():
            _, aux = model_energy(imgs)
            kp = aux["keep_prob"]

        p_e = kp / (kp.sum(dim=1, keepdim=True) + 1e-6)
        ent_e = -(p_e * torch.log(p_e + 1e-6)).sum(dim=1)
        entropy_energy.extend(ent_e.detach().cpu().numpy())

        cam = gradcam.generate(imgs)

        with torch.no_grad():
            cam_patches = model_energy.energy_layer.extract_patches(
                cam.unsqueeze(1)
            ).mean(dim=-1)

            cam_norm = cam_patches / (cam_patches.amax(dim=1, keepdim=True) + 1e-6)

            p_g = cam_norm / (cam_norm.sum(dim=1, keepdim=True) + 1e-6)
            ent_g = -(p_g * torch.log(p_g + 1e-6)).sum(dim=1)
            entropy_gradcam.extend(ent_g.cpu().numpy())

            active_energy.extend((kp > 0.5).float().mean(dim=1).cpu().numpy())
            active_gradcam.extend((cam_norm > 0.5).float().mean(dim=1).cpu().numpy())

        break

    gradcam.remove_hooks()

    return {
        "entropy_energy": float(np.mean(entropy_energy)),
        "entropy_gradcam": float(np.mean(entropy_gradcam)),
        "active_energy": float(np.mean(active_energy)),
        "active_gradcam": float(np.mean(active_gradcam)),
    }
