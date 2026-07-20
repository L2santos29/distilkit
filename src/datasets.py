"""Shared dataset registry and data-loading utilities.

Used by both ``cli.py`` (CLI mode) and ``webapp.py`` (GUI mode) to avoid
duplicating dataset definitions and download logic.
"""

import hashlib
import os
import subprocess
import tarfile
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.log_config import logger

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS = {
    "CIFAR-10": {
        "num_classes": 10,
        "in_channels": 3,
        "input_size": 32,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.247, 0.243, 0.261),
        "class_name": "CIFAR10",
        "urls": [
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "https://thor.robots.ox.ac.uk/pascal/data/cifar10/cifar-10-python.tar.gz",
        ],
        "filename": "cifar-10-python.tar.gz",
        "extracted_dir": "cifar-10-batches-py",
        "md5": "c58f30108f718f92721af3b95e74349a",
        "expected_size": 170_498_071,
    },
    "MNIST": {
        "num_classes": 10,
        "in_channels": 1,
        "input_size": 28,
        "mean": (0.1307,),
        "std": (0.3081,),
        "class_name": "MNIST",
    },
    "FashionMNIST": {
        "num_classes": 10,
        "in_channels": 1,
        "input_size": 28,
        "mean": (0.2860,),
        "std": (0.3530,),
        "class_name": "FashionMNIST",
    },
    "SVHN": {
        "num_classes": 10,
        "in_channels": 3,
        "input_size": 32,
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
        "class_name": "SVHN",
    },
}

DATASET_CHOICES = list(DATASETS.keys())
TEACHER_CHOICES = [
    "resnet18", "resnet34", "resnet50", "resnet101",
    "mobilenet_v2", "mobilenet_v3_large",
    "efficientnet_b0", "efficientnet_b1",
]
STUDENT_CHOICES = ["MiniCNN", "MiniResNet"]


def get_dataset_info(name: str) -> dict:
    """Return the metadata dict for a dataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {DATASET_CHOICES}")
    return DATASETS[name]


# ---------------------------------------------------------------------------
# CIFAR-10 optimised download
# ---------------------------------------------------------------------------


def download_cifar10(
    ds_root: str,
    cancel_flag: callable = lambda: False,
    subprocess_tracker: list | None = None,
) -> bool:
    """Download + extract CIFAR-10 with aria2c/wget/curl and exponential backoff.

    Args:
        ds_root: Directory where the dataset lives.
        cancel_flag: Callable returning True if the operation should abort.
        subprocess_tracker: Optional list to store a Popen reference for cancellation.

    Returns:
        True on success, False on failure or cancellation.
    """
    info = get_dataset_info("CIFAR-10")
    cifar_tgz = os.path.join(ds_root, info["filename"])
    extracted_dir = os.path.join(ds_root, info["extracted_dir"])
    expected_size = info["expected_size"]
    os.makedirs(ds_root, exist_ok=True)

    if os.path.isdir(extracted_dir):
        # Verify the extracted data has the expected files
        batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
        if all(os.path.isfile(os.path.join(extracted_dir, f)) for f in batch_files):
            return True
        else:
            logger.info(f"   Extracted directory incomplete, re-extracting...")
            import shutil
            shutil.rmtree(extracted_dir, ignore_errors=True)
    if os.path.exists(cifar_tgz) and os.path.getsize(cifar_tgz) == expected_size:
        logger.info("   Extracting previously downloaded file...")
        with tarfile.open(cifar_tgz, "r:gz") as tar:
            tar.extractall(path=ds_root, filter='data')
        logger.info("   ✅ CIFAR-10 ready!")
        return True
    if os.path.exists(cifar_tgz):
        logger.info(f"   Removing partial download ({os.path.getsize(cifar_tgz)/1e6:.0f} MB)...")
        os.remove(cifar_tgz)

    logger.info("⬇️ Downloading CIFAR-10 (170 MB)...")

    has_aria2c = subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0
    has_wget = subprocess.run(["which", "wget"], capture_output=True).returncode == 0
    has_curl = subprocess.run(["which", "curl"], capture_output=True).returncode == 0

    max_retries = 3
    base_delay = 2
    downloaded_ok = False

    for attempt in range(1, max_retries + 1):
        if cancel_flag():
            return False

        if attempt > 1:
            import time as _time
            delay = base_delay * (2 ** (attempt - 2))
            logger.info(f"   Retry {attempt}/{max_retries} in {delay}s...")
            _time.sleep(delay)
            if cancel_flag():
                return False
            if os.path.exists(cifar_tgz):
                os.remove(cifar_tgz)

        downloaded_ok = False
        for url in info["urls"]:
            if cancel_flag() or downloaded_ok:
                break

            tools: list[list[str]] = []
            if has_aria2c:
                tools.append(["aria2c", "-x", "4", "-s", "4", "-d", ds_root, "-o", info["filename"]])
            if has_wget:
                tools.append(["wget", "-O", cifar_tgz, "--show-progress"])
            if has_curl:
                tools.append(["curl", "-#", "-Lo", cifar_tgz])

            for cmd_template in tools:
                if cancel_flag():
                    return False
                tool_name = cmd_template[0]
                logger.info(f"   Trying {tool_name}...")
                cmd = cmd_template + [url]
                proc = subprocess.Popen(cmd)
                if subprocess_tracker is not None:
                    subprocess_tracker.append(proc)
                proc.wait()
                if subprocess_tracker is not None:
                    subprocess_tracker.clear()
                if os.path.exists(cifar_tgz) and os.path.getsize(cifar_tgz) == expected_size:
                    downloaded_ok = True
                    break
                if cancel_flag():
                    return False

        if not downloaded_ok:
            import urllib.request
            logger.info("   Trying Python (fallback)...")
            for url in info["urls"]:
                if cancel_flag():
                    return False
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    resp = urllib.request.urlopen(req)
                    total = int(resp.headers.get("Content-Length", expected_size))
                    downloaded, chunk = 0, 8192
                    with open(cifar_tgz, "wb") as f:
                        while True:
                            if cancel_flag():
                                return False
                            data = resp.read(chunk)
                            if not data:
                                break
                            f.write(data)
                            downloaded += len(data)
                            if downloaded % (chunk * 200) == 0:
                                logger.info(f"   {min(downloaded/total*100,100):.0f}%")
                    if os.path.getsize(cifar_tgz) == expected_size:
                        downloaded_ok = True
                        break
                except Exception as e:
                    logger.info(f"   Error: {e}")
                    continue

        if downloaded_ok:
            break

    if not downloaded_ok:
        logger.info("❌ All retries exhausted. Could not download from any server.")
        return False

    md5_actual = hashlib.md5(open(cifar_tgz, "rb").read()).hexdigest()
    if md5_actual != info["md5"]:
        logger.info("⚠️  MD5 mismatch — file corrupted, will retry next time.")
        os.remove(cifar_tgz)
        return False

    logger.info("   Extracting...")
    with tarfile.open(cifar_tgz, "r:gz") as tar:
        tar.extractall(path=ds_root, filter='data')
    logger.info("✅ CIFAR-10 ready!")
    return True


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _check_torchvision_dataset(name: str, root: str) -> bool:
    """Check if a torchvision dataset's raw/processed files exist."""
    if os.path.isdir(os.path.join(root, "raw")) and os.listdir(os.path.join(root, "raw")):
        return True
    if os.path.isdir(os.path.join(root, "processed")) and os.listdir(os.path.join(root, "processed")):
        return True
    return False


def get_dataset_loaders(
    dataset_name: str,
    batch_size: int,
    data_root: str = "./data",
    cancel_flag: callable = lambda: False,
    subprocess_tracker: list | None = None,
) -> tuple[DataLoader, DataLoader, int, int] | None:
    """Return ``(train_loader, val_loader, num_classes, in_channels)``.

    Returns ``None`` if preparation fails.
    """
    info = get_dataset_info(dataset_name)
    ds_class = getattr(datasets, info["class_name"])
    num_classes = info["num_classes"]
    in_channels = info["in_channels"]
    input_size = info["input_size"]
    mean, std = info["mean"], info["std"]
    ds_root = os.path.join(data_root, dataset_name)
    os.makedirs(ds_root, exist_ok=True)

    # Download / verify
    try:
        if dataset_name == "CIFAR-10":
            ok = download_cifar10(ds_root, cancel_flag, subprocess_tracker)
            if not ok:
                return None
            extracted = os.path.join(ds_root, info["extracted_dir"])
            if not os.path.isdir(extracted):
                logger.info("❌ CIFAR-10 data not found after download.")
                return None
        else:
            has_files = _check_torchvision_dataset(dataset_name, ds_root)
            if not has_files:
                logger.info(f"⬇️ Downloading {dataset_name}...")
                # Trigger download by instantiating with download=True
                if dataset_name == "SVHN":
                    ds_class(root=ds_root, split="train", download=True)
                else:
                    ds_class(root=ds_root, train=True, download=True)
    except (OSError, IOError, RuntimeError) as e:
        logger.info(f"❌ Dataset I/O error: {e}")
        return None

    # Transforms
    if input_size <= 32:
        train_transform = transforms.Compose([
            transforms.RandomCrop(input_size, padding=4 if input_size >= 28 else 2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load
    try:
        if dataset_name == "SVHN":
            train_set = ds_class(root=ds_root, split="train", download=False, transform=train_transform)
            val_set = ds_class(root=ds_root, split="test", download=False, transform=val_transform)
        else:
            train_set = ds_class(root=ds_root, train=True, download=False, transform=train_transform)
            val_set = ds_class(root=ds_root, train=False, download=False, transform=val_transform)
    except (OSError, RuntimeError) as e:
        # Safety fallback: if loading fails, try with download=True
        logger.info(f"   First load failed ({e}), retrying with download=True...")
        try:
            if dataset_name == "SVHN":
                train_set = ds_class(root=ds_root, split="train", download=True, transform=train_transform)
                val_set = ds_class(root=ds_root, split="test", download=True, transform=val_transform)
            else:
                train_set = ds_class(root=ds_root, train=True, download=True, transform=train_transform)
                val_set = ds_class(root=ds_root, train=False, download=True, transform=val_transform)
        except (OSError, RuntimeError) as e2:
            logger.info(f"❌ Failed to load dataset: {e2}")
            return None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, num_classes, in_channels
