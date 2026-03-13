# ERSM — Energy-based Redundancy & Saliency Masking

Learnable patch-level feature masking for fine-grained image classification.
ERSM inserts a lightweight energy layer between a frozen CNN backbone and the
classifier head, learning which spatial patches to keep or suppress during
training.

## Project Structure

```
ERSM/
├── main.py              # Entry point — runs the full experiment grid
├── config.py            # All hyperparameters and experiment grid definition
├── requirements.txt
├── .gitignore
├── README.md
└── src/
    ├── __init__.py      # Public API re-exports
    ├── data.py          # Dataset loading (CUB-200, Oxford Pets, Food-101)
    ├── models.py        # EnergyMaskLayer, FrozenBackboneWrapper, build_backbone
    ├── training.py      # Training loop and accuracy evaluation
    ├── analysis.py      # Robustness curves, mask distribution, visual panels, failure modes
    └── gradcam.py       # Grad-CAM, deletion AUC comparison, sparsity/entropy metrics
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and a CUDA-capable GPU (falls back to CPU automatically).

## Quick Start

```bash
# Run all experiments defined in config.py
python main.py
```

Results (JSON + PNG visualisations) are saved to the `results/` directory.

## Configuration

Edit `config.py` to change:

| Parameter       | Description                              | Default              |
|-----------------|------------------------------------------|----------------------|
| `DATASET_NAME`  | Dataset to use (`cub200`, `pets`, `food`) | `food`               |
| `ARCHS`         | Backbone architectures to benchmark      | ResNet-50, ConvNeXt-T, EfficientNetV2-S |
| `EPOCHS`        | Training epochs                          | 20                   |
| `LAMBDA`        | Unary energy weight                      | 1e-3                 |
| `GAMMA`         | Pairwise smoothness weight               | 1e-3                 |
| `TEMP`          | Sigmoid temperature for keep-probability | 1.0                  |

The experiment grid sweeps over `SIZES` (input resolutions) and `PATCHES`
(patch sizes applied to feature maps).

## Method

The **EnergyMaskLayer** operates on the feature map produced by a frozen
backbone.  It unfolds the feature map into spatial patches and computes:

1. **Unary energy** — a learned linear projection per patch, biased toward
   keeping all patches initially.
2. **Pairwise energy** — a smoothness penalty over the 8-connected
   neighbourhood, encouraging spatially coherent masks.

The keep-probability is derived via a sigmoid over the combined energy score.
During training, the energy regularisation term steers the mask toward sparse
but discriminative feature selection.

## Supported Datasets

- **CUB-200-2011** — 200 bird species (auto-downloaded)
- **Oxford-IIIT Pets** — 37 breeds (auto-downloaded)
- **Food-101** — 101 food categories (auto-downloaded)

## Supported Architectures

ResNet-18, ResNet-50, ConvNeXt (tiny/small/base/large),
EfficientNet-B0–B7, EfficientNetV2 (S/M/L).

