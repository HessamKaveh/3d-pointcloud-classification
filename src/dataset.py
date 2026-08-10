import os
import glob
import zipfile
import requests
import h5py
import numpy as np
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from torch.utils.data import Dataset, DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

NUM_POINTS = 1024
NUM_CLASSES = 40

ARCHIVE_NAME = "modelnet40_ply_hdf5_2048.zip"
ARCHIVE_PATH = os.path.join(DATA_DIR, ARCHIVE_NAME)
PARTIAL_ARCHIVE_PATH = ARCHIVE_PATH + ".part"
MODELNET_DIR = os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048")

# Primary PointNet dataset location + a public mirror of the same archive.
MODELNET40_URLS = [
    "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
    "https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip?download=true",
]


def _make_session():
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def _download_file(url, destination):
    session = _make_session()

    with session.get(
        url,
        stream=True,
        timeout=(15, 120),   # connect timeout, read timeout
        allow_redirects=True,
    ) as response:
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    pct = 100.0 * downloaded / total
                    print(
                        f"\rDownloaded {downloaded / (1024**2):.1f} MB "
                        f"({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\rDownloaded {downloaded / (1024**2):.1f} MB",
                        end="",
                        flush=True,
                    )
        print()


def _archive_is_valid(path):
    if not os.path.isfile(path):
        return False

    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad_file = zf.testzip()
            return bad_file is None
    except zipfile.BadZipFile:
        return False


def download_modelnet40():
    """Download and extract the preprocessed ModelNet40 HDF5 point clouds."""
    if os.path.isdir(MODELNET_DIR):
        return MODELNET_DIR

    os.makedirs(DATA_DIR, exist_ok=True)

    # Remove a stale/incomplete temporary download from an earlier failed run.
    if os.path.exists(PARTIAL_ARCHIVE_PATH):
        os.remove(PARTIAL_ARCHIVE_PATH)

    # If a manually downloaded archive already exists, validate and use it.
    if os.path.isfile(ARCHIVE_PATH) and not _archive_is_valid(ARCHIVE_PATH):
        print("Existing ModelNet40 zip is incomplete/corrupt; removing it.")
        os.remove(ARCHIVE_PATH)

    if not os.path.isfile(ARCHIVE_PATH):
        errors = []

        for i, url in enumerate(MODELNET40_URLS, start=1):
            print(f"Downloading ModelNet40 (~416 MB), source {i}/{len(MODELNET40_URLS)}...")
            try:
                _download_file(url, PARTIAL_ARCHIVE_PATH)

                if not _archive_is_valid(PARTIAL_ARCHIVE_PATH):
                    raise RuntimeError("Downloaded file is not a valid ZIP archive.")

                os.replace(PARTIAL_ARCHIVE_PATH, ARCHIVE_PATH)
                print("Download complete.")
                break

            except Exception as exc:
                errors.append(f"{url}\n  -> {type(exc).__name__}: {exc}")
                if os.path.exists(PARTIAL_ARCHIVE_PATH):
                    os.remove(PARTIAL_ARCHIVE_PATH)

                if i < len(MODELNET40_URLS):
                    print("This source failed. Trying the fallback mirror...")

        if not os.path.isfile(ARCHIVE_PATH):
            details = "\n\n".join(errors)
            raise RuntimeError(
                "Could not download ModelNet40 from any configured source.\n\n"
                f"{details}\n\n"
                "You can also download modelnet40_ply_hdf5_2048.zip manually "
                f"and place it here:\n{ARCHIVE_PATH}"
            )

    print("Extracting ModelNet40...")
    try:
        with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"The dataset archive is corrupt: {ARCHIVE_PATH}. "
            "Delete it and run train.py again."
        ) from exc

    if not os.path.isdir(MODELNET_DIR):
        raise RuntimeError(
            f"Dataset archive was extracted, but {MODELNET_DIR} was not found."
        )

    return MODELNET_DIR


def _load_h5_files(split):
    dataset_dir = download_modelnet40()
    pattern = os.path.join(dataset_dir, f"ply_data_{split}*.h5")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No HDF5 files found for split='{split}' in {dataset_dir}"
        )

    data_parts = []
    label_parts = []

    for file_path in files:
        with h5py.File(file_path, "r") as f:
            data_parts.append(f["data"][:].astype(np.float32))
            label_parts.append(f["label"][:].astype(np.int64).reshape(-1))

    points = np.concatenate(data_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    return points, labels


class ModelNet40Dataset(Dataset):
    """ModelNet40 point-cloud classification dataset."""

    def __init__(self, split="train", num_points=NUM_POINTS, augment=False):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        if not 1 <= num_points <= 2048:
            raise ValueError("num_points must be between 1 and 2048")

        self.split = split
        self.num_points = num_points
        self.augment = augment and split == "train"
        self.points, self.labels = _load_h5_files(split)

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _augment(points):
        # Random rotation around the vertical Y axis.
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array(
            [[c, 0.0, s],
             [0.0, 1.0, 0.0],
             [-s, 0.0, c]],
            dtype=np.float32,
        )
        points = points @ rotation.T

        # Small Gaussian jitter.
        jitter = np.clip(
            np.random.normal(0.0, 0.01, size=points.shape),
            -0.05,
            0.05,
        ).astype(np.float32)
        return points + jitter

    def __getitem__(self, idx):
        points = self.points[idx, : self.num_points].copy()
        label = int(self.labels[idx])

        if self.augment:
            points = self._augment(points)

        return (
            torch.from_numpy(points).float(),
            torch.tensor(label, dtype=torch.long),
        )


def get_dataloaders(
    batch_size=32,
    val_fraction=0.1,
    num_workers=2,
    seed=42,
):
    """
    Return train/validation loaders.

    Validation is a deterministic held-out subset of the official training split.
    The official ModelNet40 test split is reserved for evaluate.py.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    base_dataset = ModelNet40Dataset(split="train", augment=False)

    n_total = len(base_dataset)
    n_val = max(1, int(round(n_total * val_fraction)))

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()

    train_dataset = ModelNet40Dataset(split="train", augment=True)
    val_dataset = base_dataset

    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=32, num_workers=2):
    test_dataset = ModelNet40Dataset(split="test", augment=False)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def get_class_names():
    dataset_dir = download_modelnet40()
    path = os.path.join(dataset_dir, "shape_names.txt")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
