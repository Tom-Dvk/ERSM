import torch

# ===============================================================
# Global Configuration
# ===============================================================
GLOBAL_CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # Dataset Selection
    "DATASET_NAME": "cub200",
    "DATA_ROOT": "./data",

    # Results
    "RESULTS_DIR": "./results",
    "JSON_PATH": "./results/all_experiments.json",

    # Training
    "BATCH_SIZE": 32,
    "EPOCHS": 20,
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "SEEDS": [0],

    # ERSM Energy Hyperparameters
    "LAMBDA": 1e-3,
    "GAMMA": 1e-3,
    "TEMP": 1.0,

    # Architectures to benchmark
    "ARCHS": [
        "resnet50",
        "convnext_tiny",
        "efficientnet_v2_s",
    ],
}

# ===============================================================
# Experiment Grid: arch x input_size x patch_size
# ===============================================================
SIZES = [224, 256, 448]
PATCHES = [1, 2]

EXPERIMENTS = [
    {"arch": a, "input_size": s, "patch_size": p}
    for a in GLOBAL_CONFIG["ARCHS"]
    for s in SIZES
    for p in PATCHES
]
