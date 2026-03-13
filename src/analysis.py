import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


@torch.no_grad()
def analyze_mask_distribution(model, loader, device, out_path):
    """Plot a histogram of patch keep-probabilities and return the mean."""
    model.eval()
    all_probs = []
    print("   -> Computing Mask Distribution...")

    count = 0
    for imgs, _ in loader:
        if count > 500:
            break
        imgs = imgs.to(device)
        _, aux = model(imgs)
        all_probs.append(aux["keep_prob"].cpu().numpy().flatten())
        count += imgs.size(0)

    data = np.concatenate(all_probs)
    mean_val = float(data.mean())

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=50, range=(0, 1), color="teal", alpha=0.7, edgecolor="black")
    plt.axvline(mean_val, color="red", linestyle="dashed", linewidth=1,
                label=f"Mean: {mean_val:.2f}")
    plt.title("Distribution of Patch Keep Probabilities")
    plt.xlabel("Keep Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()

    return mean_val


@torch.no_grad()
def evaluate_robustness_granular(model, loader, device, out_path_curve):
    """Deletion curves: Energy vs Magnitude vs Random patch removal."""
    model.eval()

    imgs, labels = next(iter(loader))
    imgs = imgs.to(device)
    _, aux = model(imgs)
    N = aux["num_patches"]

    print(f"   -> Running Granular Deletion (0 to {N} patches) "
          f"[Energy vs Magnitude vs Random]...")

    correct_energy = {k: 0 for k in range(N + 1)}
    correct_magnitude = {k: 0 for k in range(N + 1)}
    correct_random = {k: 0 for k in range(N + 1)}
    total_samples = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        B = imgs.size(0)
        total_samples += B

        feat = model.forward_features_pre_mask(imgs)
        _, aux = model(imgs)
        z = aux["z"]

        idx_ranked = torch.argsort(z, dim=1, descending=True)

        patches = model.energy_layer.extract_patches(feat)
        patch_norms = patches.norm(p=2, dim=-1)
        idx_magnitude = torch.argsort(patch_norms, dim=1, descending=False)

        idx_random = torch.stack(
            [torch.randperm(N, device=device) for _ in range(B)]
        )

        for k in range(N + 1):
            mask_e = torch.ones(B, N, device=device)
            if k > 0:
                mask_e.scatter_(1, idx_ranked[:, :k], 0.0)
            logits_e, _ = model(imgs, forced_mask=mask_e, scale_compensation=False)
            correct_energy[k] += (logits_e.argmax(1) == labels).sum().item()

            mask_m = torch.ones(B, N, device=device)
            if k > 0:
                mask_m.scatter_(1, idx_magnitude[:, :k], 0.0)
            logits_m, _ = model(imgs, forced_mask=mask_m, scale_compensation=False)
            correct_magnitude[k] += (logits_m.argmax(1) == labels).sum().item()

            mask_r = torch.ones(B, N, device=device)
            if k > 0:
                mask_r.scatter_(1, idx_random[:, :k], 0.0)
            logits_r, _ = model(imgs, forced_mask=mask_r, scale_compensation=False)
            correct_random[k] += (logits_r.argmax(1) == labels).sum().item()

    steps = list(range(N + 1))
    acc_e = [100.0 * correct_energy[k] / total_samples for k in steps]
    acc_m = [100.0 * correct_magnitude[k] / total_samples for k in steps]
    acc_r = [100.0 * correct_random[k] / total_samples for k in steps]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, acc_e, marker=".", linestyle="-", color="red",
             label="Energy Deletion")
    plt.plot(steps, acc_m, marker="", linestyle="-.", color="green",
             label="Magnitude Deletion")
    plt.plot(steps, acc_r, marker="", linestyle="--", color="blue",
             label="Random Deletion")
    plt.title(f"Robustness: Accuracy vs Patches Deleted (Max {N})")
    plt.xlabel("Patches Deleted")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path_curve)
    plt.close()

    return steps, acc_e, acc_m, acc_r


@torch.no_grad()
def save_visual_panels(model, loader, device, out_dir):
    """Save per-sample panels: original | keep-prob heatmap | faded image."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs[:10].to(device)

    feat = model.forward_features_pre_mask(imgs)
    _, aux = model.energy_layer(feat)
    kp = aux["keep_prob"]
    H, W = aux["grid_shape"]

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    imgs_disp = (imgs * std + mean).clamp(0, 1)

    for i in range(len(imgs)):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(imgs_disp[i].permute(1, 2, 0).cpu().numpy())
        axs[0].set_title("Input")
        axs[0].axis("off")

        mask = kp[i].view(H, W).cpu().numpy()
        axs[1].imshow(mask, cmap="inferno", vmin=0, vmax=1)
        axs[1].set_title(f"Keep Prob (Mean: {mask.mean():.2f})")
        axs[1].axis("off")

        mask_t = kp[i].view(1, 1, H, W)
        mask_up = F.interpolate(mask_t, size=imgs.shape[2:],
                                mode="bilinear", align_corners=False)
        masked_img = imgs_disp[i] * mask_up[0]
        axs[2].imshow(masked_img.permute(1, 2, 0).cpu().numpy())
        axs[2].set_title("Faded Image")
        axs[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sample_{i}.png"))
        plt.close()


@torch.no_grad()
def compare_and_visualize_improvements(model_base, model_energy, loader, device,
                                       out_dir, class_names=None, num_examples=10):
    """Find images where baseline fails but ERSM succeeds and save panels."""
    print("   -> Generating Baseline vs Energy Comparison...")
    os.makedirs(out_dir, exist_ok=True)
    model_base.eval()
    model_energy.eval()
    found = 0

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for imgs, labels in loader:
        if found >= num_examples:
            break
        imgs, labels = imgs.to(device), labels.to(device)

        logits_b, _ = model_base(imgs)
        pred_b = logits_b.argmax(1)

        logits_e, aux = model_energy(imgs)
        pred_e = logits_e.argmax(1)

        mask_imp = (pred_b != labels) & (pred_e == labels)
        idxs = mask_imp.nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
            continue

        kp = aux["keep_prob"]
        H, W = aux["grid_shape"]

        for idx in idxs:
            if found >= num_examples:
                break
            i = idx.item()

            img_disp = (imgs[i].unsqueeze(0) * std + mean).clamp(0, 1)
            mask_small = kp[i].view(1, 1, H, W)
            mask_up = F.interpolate(mask_small, size=img_disp.shape[2:],
                                    mode="bilinear", align_corners=False).clamp(0, 1)
            img_faded = img_disp * mask_up

            t_lbl = class_names[labels[i]] if class_names else str(labels[i].item())
            b_lbl = class_names[pred_b[i]] if class_names else str(pred_b[i].item())
            e_lbl = class_names[pred_e[i]] if class_names else str(pred_e[i].item())

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img_disp[0].permute(1, 2, 0).cpu().numpy())
            axs[0].set_title(f"True: {t_lbl}")
            axs[0].axis("off")

            axs[1].imshow(img_disp[0].permute(1, 2, 0).cpu().numpy())
            axs[1].text(10, 30, "FAIL", color="red", fontsize=12,
                        fontweight="bold", backgroundcolor="white")
            axs[1].set_title(f"Baseline: {b_lbl}\n(Wrong)")
            axs[1].axis("off")

            axs[2].imshow(img_faded[0].permute(1, 2, 0).cpu().numpy())
            axs[2].set_title(f"Energy: {e_lbl}\n(Correct)")
            axs[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"improvement_{found}.png"))
            plt.close()
            found += 1


@torch.no_grad()
def analyze_failure_modes(model, loader, device, out_dir,
                          class_names=None, top_k=5):
    """Categorise ERSM failures into focused / distracted / high-confidence."""
    print("   -> Running Failure Analysis...")
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    failures = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, aux = model(imgs)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(1)

        wrong_idx = (preds != labels).nonzero(as_tuple=True)[0]
        if len(wrong_idx) == 0:
            continue

        kp = aux["keep_prob"]
        H, W = aux["grid_shape"]

        for idx in wrong_idx:
            i = idx.item()
            conf = probs[i, preds[i]].item()
            sparsity = 1.0 - kp[i].mean().item()

            failures.append({
                "img": imgs[i],
                "label": labels[i].item(),
                "pred": preds[i].item(),
                "conf": conf,
                "sparsity": sparsity,
                "keep_prob": kp[i].view(1, 1, H, W),
            })

    if len(failures) == 0:
        return

    focused = sorted(failures, key=lambda x: x["sparsity"], reverse=True)[:top_k]
    distracted = sorted(failures, key=lambda x: x["sparsity"], reverse=False)[:top_k]
    confident = sorted(failures, key=lambda x: x["conf"], reverse=True)[:top_k]

    def _plot_batch(fail_list, mode_name):
        save_path = os.path.join(out_dir, mode_name)
        os.makedirs(save_path, exist_ok=True)
        for i, item in enumerate(fail_list):
            img_disp = (item["img"].unsqueeze(0) * std + mean).clamp(0, 1)
            mask_up = F.interpolate(
                item["keep_prob"], size=img_disp.shape[2:],
                mode="bilinear", align_corners=False,
            ).clamp(0, 1)
            masked_img = img_disp * mask_up

            t_lbl = class_names[item["label"]] if class_names else str(item["label"])
            p_lbl = class_names[item["pred"]] if class_names else str(item["pred"])

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img_disp[0].permute(1, 2, 0).cpu().numpy())
            axs[0].set_title(f"True: {t_lbl}")
            axs[0].axis("off")
            axs[1].imshow(item["keep_prob"][0, 0].cpu().numpy(),
                          cmap="inferno", vmin=0, vmax=1)
            axs[1].set_title(f"Mask (Sparsity: {item['sparsity']:.2f})")
            axs[1].axis("off")
            axs[2].imshow(masked_img[0].permute(1, 2, 0).cpu().numpy())
            axs[2].set_title(f"Pred: {p_lbl} ({item['conf']:.2f})")
            axs[2].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"fail_{i}.png"))
            plt.close()

    _plot_batch(focused, "focused_failures")
    _plot_batch(distracted, "distracted_failures")
    _plot_batch(confident, "high_conf_failures")
