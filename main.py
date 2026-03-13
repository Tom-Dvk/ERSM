"""
ERSM — Energy-based Redundancy & Saliency Masking
Main experiment runner: trains Energy + Baseline models across the full
architecture x input-size x patch-size grid defined in config.py.
"""

import os
import json
import traceback

import torch

from config import GLOBAL_CONFIG, EXPERIMENTS
from src.data import set_seed, get_dataloaders
from src.models import FrozenBackboneWrapper, get_feature_hw
from src.training import train_engine, evaluate_accuracy
from src.analysis import (
    analyze_mask_distribution,
    evaluate_robustness_granular,
    save_visual_panels,
    compare_and_visualize_improvements,
    analyze_failure_modes,
)
from src.gradcam import deletion_auc_compare, compare_sparsity_entropy


def run_single_experiment(cfg, run_id):
    """Train Energy + Baseline models for one (arch, input_size, patch_size) config."""
    device = GLOBAL_CONFIG["DEVICE"]
    print(f"\n{'=' * 40}\nRunning Experiment: {run_id}\nConfig: {cfg}\n{'=' * 40}")

    # 1) Feature-map / patch compatibility
    Hf, Wf = get_feature_hw(cfg["arch"], device, cfg["input_size"])
    if (Hf % cfg["patch_size"] != 0) or (Wf % cfg["patch_size"] != 0):
        print(f"!!! SKIPPING: arch={cfg['arch']} feature={Hf}x{Wf} "
              f"not divisible by patch={cfg['patch_size']}")
        return None

    # 2) Data
    train_dl, test_dl, n_cls = get_dataloaders(
        GLOBAL_CONFIG["DATASET_NAME"], GLOBAL_CONFIG["DATA_ROOT"],
        cfg["input_size"], GLOBAL_CONFIG["BATCH_SIZE"],
    )

    class_names = None
    ds = test_dl.dataset
    if hasattr(ds, "classes"):
        class_names = ds.classes
    elif hasattr(ds, "_classes"):
        class_names = ds._classes

    # 3) Train Energy model
    model = FrozenBackboneWrapper(
        cfg["arch"], n_cls, use_energy=True,
        patch_size=cfg["patch_size"],
        lambda_u=GLOBAL_CONFIG["LAMBDA"],
        gamma_p=GLOBAL_CONFIG["GAMMA"],
        temperature=GLOBAL_CONFIG["TEMP"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=GLOBAL_CONFIG["LR"],
        weight_decay=GLOBAL_CONFIG["WEIGHT_DECAY"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=GLOBAL_CONFIG["EPOCHS"],
    )

    energy_history = train_engine(
        model, train_dl, test_dl, opt, scheduler,
        device, GLOBAL_CONFIG["EPOCHS"], name=f"Energy[{cfg['arch']}]",
    )

    # 4) Train Baseline model
    model_base = FrozenBackboneWrapper(
        cfg["arch"], n_cls, use_energy=False,
    ).to(device)

    opt_base = torch.optim.AdamW(
        model_base.parameters(), lr=GLOBAL_CONFIG["LR"],
        weight_decay=GLOBAL_CONFIG["WEIGHT_DECAY"],
    )
    scheduler_base = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_base, T_max=GLOBAL_CONFIG["EPOCHS"],
    )

    baseline_history = train_engine(
        model_base, train_dl, test_dl, opt_base, scheduler_base,
        device, GLOBAL_CONFIG["EPOCHS"], name=f"Baseline[{cfg['arch']}]",
    )

    # 5) Analyse & save
    out_dir = os.path.join(GLOBAL_CONFIG["RESULTS_DIR"], run_id)
    os.makedirs(out_dir, exist_ok=True)

    mean_keep = analyze_mask_distribution(
        model, test_dl, device, os.path.join(out_dir, "mask_dist.png"),
    )
    steps, acc_e, acc_m, acc_r = evaluate_robustness_granular(
        model, test_dl, device, os.path.join(out_dir, "robustness_curve.png"),
    )

    print("   -> Generating Visual Panels...")
    save_visual_panels(model, test_dl, device, out_dir)

    compare_and_visualize_improvements(
        model_base, model, test_dl, device,
        out_dir=os.path.join(out_dir, "improvements"),
        class_names=class_names,
    )
    analyze_failure_modes(
        model, test_dl, device,
        out_dir=os.path.join(out_dir, "failures"),
        class_names=class_names,
    )

    gradcam_metrics = deletion_auc_compare(
        model, model_base, test_dl, device,
        out_path_curve=os.path.join(out_dir, "gradcam_deletion_curve.png"),
    )
    sparsity_metrics = compare_sparsity_entropy(
        model, model_base, test_dl, device,
    )

    result = {
        "config": cfg,
        "feature_hw": [int(Hf), int(Wf)],
        "energy_model": {
            "mean_keep_prob_final": mean_keep,
            "robustness_curve": {
                "steps": steps,
                "acc_energy": acc_e,
                "acc_magnitude": acc_m,
                "acc_random": acc_r,
            },
            "gradcam_comparison": gradcam_metrics,
            "sparsity_comparison": sparsity_metrics,
            "history": energy_history,
        },
        "baseline_model": {
            "history": baseline_history,
        },
    }

    per_run_json = os.path.join(out_dir, "result.json")
    with open(per_run_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"   -> Saved {per_run_json}")

    return result


def main():
    set_seed(GLOBAL_CONFIG["SEEDS"][0])
    os.makedirs(GLOBAL_CONFIG["RESULTS_DIR"], exist_ok=True)

    all_results = {}

    for exp_cfg in EXPERIMENTS:
        run_id = (f"{exp_cfg['arch']}_in{exp_cfg['input_size']}"
                  f"_patch{exp_cfg['patch_size']}")
        try:
            res = run_single_experiment(exp_cfg, run_id)
            if res is not None:
                all_results[run_id] = res
        except Exception as e:
            print(f"Error in {run_id}: {e}")
            traceback.print_exc()

    with open(GLOBAL_CONFIG["JSON_PATH"], "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone! Results saved to {GLOBAL_CONFIG['JSON_PATH']}")


if __name__ == "__main__":
    main()
