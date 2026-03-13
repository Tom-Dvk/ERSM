import os
import random
import tarfile
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder, Food101, OxfordIIITPet
from torchvision.datasets.utils import download_url


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def download_and_prepare_cub(root):
    """Download CUB-200-2011 and reorganise into train/test ImageFolder layout."""
    dataset_name = "CUB_200_2011"
    cub_root = os.path.join(root, dataset_name)
    tgz_path = os.path.join(root, f"{dataset_name}.tgz")

    if os.path.exists(os.path.join(cub_root, "train")):
        return cub_root

    url = "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz"
    if not os.path.exists(tgz_path) and not os.path.exists(cub_root):
        print("Downloading CUB-200-2011...")
        try:
            download_url(url, root, filename=f"{dataset_name}.tgz")
        except Exception:
            pass

    if not os.path.exists(os.path.join(cub_root, "images")):
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=root)

    nested = os.path.join(root, "CUB_200_2011", "CUB_200_2011")
    if os.path.exists(nested):
        for item in os.listdir(nested):
            shutil.move(os.path.join(nested, item), cub_root)
        os.rmdir(nested)

    images_txt = os.path.join(cub_root, "images.txt")
    split_txt = os.path.join(cub_root, "train_test_split.txt")
    if not os.path.exists(images_txt):
        return cub_root

    id2path = {}
    with open(images_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            id2path[parts[0]] = parts[1]

    id2train = {}
    with open(split_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            id2train[parts[0]] = int(parts[1])

    os.makedirs(os.path.join(cub_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(cub_root, "test"), exist_ok=True)
    src_dir = os.path.join(cub_root, "images")

    for idx, rel_path in id2path.items():
        split = "train" if id2train.get(idx, 0) == 1 else "test"
        src = os.path.join(src_dir, rel_path)
        dst = os.path.join(cub_root, split, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src):
            shutil.move(src, dst)
    return cub_root


def get_dataloaders(name, root, input_size, batch_size):
    """Build train/test DataLoaders for the specified dataset.

    Supported datasets: ``cub200``, ``pets``, ``food`` (Food-101).
    """
    print(f"Loading {name} @ {input_size}x{input_size}...")
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    os.makedirs(root, exist_ok=True)

    if name == "cub200":
        cub_root = download_and_prepare_cub(root)
        train_ds = ImageFolder(os.path.join(cub_root, "train"), transform=train_tf)
        test_ds = ImageFolder(os.path.join(cub_root, "test"), transform=test_tf)
        num_classes = 200

    elif name == "pets":
        train_ds = OxfordIIITPet(
            root=root, split="trainval", target_types="category",
            download=True, transform=train_tf,
        )
        test_ds = OxfordIIITPet(
            root=root, split="test", target_types="category",
            download=True, transform=test_tf,
        )
        num_classes = 37
        print(f"   -> Oxford Pets: train={len(train_ds)} test={len(test_ds)} classes={num_classes}")

    elif name in ("food", "food101", "food-101"):
        food_root = os.path.join(root, "food101")
        train_ds = Food101(root=food_root, split="train", download=True, transform=train_tf)
        test_ds = Food101(root=food_root, split="test", download=True, transform=test_tf)
        num_classes = 101
        print("   -> Food-101 classes:", num_classes, flush=True)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    return train_loader, test_loader, num_classes
