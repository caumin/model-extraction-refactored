"""
Module for loading and preprocessing various datasets."""

from typing import Tuple, List, Optional, Sequence
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms as T
from PIL import Image
import pickle
from collections import abc

def windowsafe_loader(num_workers=0, pin_memory=False):
    """Returns a dictionary of DataLoader arguments for Windows compatibility."""
    return dict(num_workers=num_workers, pin_memory=pin_memory)

from torchvision.datasets import ImageFolder
import os

class CleanImageFolder(ImageFolder):
    """A custom ImageFolder dataset that excludes hidden/checkpoint folders."""
    def find_classes(self, directory):
        # Exclude hidden/checkpoint folders
        ignore = {'.ipynb_checkpoints'}
        classes = [
            d.name for d in os.scandir(directory)
            if d.is_dir() and not d.name.startswith('.') and d.name not in ignore
        ]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


# -------- Normalize Utilities --------
def _broadcast_stats(stats, in_ch: int):
    """Broadcasts mean/std statistics to match the number of input channels."""
    if stats is None:
        return None
    stats = tuple(stats)
    if len(stats) == 1 and in_ch > 1:
        return tuple([stats[0]] * in_ch)
    if len(stats) == in_ch:
        return stats
    if in_ch == 1 and len(stats) == 3:
        # If grayscale but 3-channel args, average to single value
        val = float(np.mean(stats))
        return (val,)
    raise ValueError(f"Mean/Std length {len(stats)} not compatible with in_ch={in_ch}")

def _make_norm_transform(normalize: Optional[dict], in_ch: int):
    """Creates a normalization transform based on the provided normalization dictionary."""
    if not normalize or normalize.get('mode', 'none') == 'none':
        return None
    mode = normalize.get('mode')
    if mode == 'imagenet':
        mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    elif mode == 'mnist':
        # Commonly used official MNIST stats
        mean, std = (0.1307,), (0.3081,)
    elif mode == 'half':
        mean, std = (0.5,), (0.5,)
    elif mode == 'custom':
        mean = normalize.get('mean'); std = normalize.get('std')
        if mean is None or std is None:
            raise ValueError("normalize.mode=custom requires mean and std")
    else:
        return None
    mean = _broadcast_stats(mean, in_ch)
    std  = _broadcast_stats(std,  in_ch)
    return T.Normalize(mean, std)

# -------- MNIST --------
def get_mnist_loaders(batch_size=128, img_size=28, root='./data', download=True,
                      num_workers=0, pin_memory=False, as_rgb=False, normalize: Optional[dict]=None, in_ch: int=1):
    """Returns DataLoader instances for the MNIST dataset."""
    tfm_train = [T.Resize((img_size, img_size))]
    tfm_test  = [T.Resize((img_size, img_size))]
    if as_rgb:
        tfm_train.insert(0, T.Grayscale(num_output_channels=3))
        tfm_test .insert(0, T.Grayscale(num_output_channels=3))
        in_ch_eff = 3
    else:
        in_ch_eff = in_ch
    tfm_train.append(T.ToTensor()); tfm_test.append(T.ToTensor())
    norm = _make_norm_transform(normalize, in_ch_eff)
    if norm is not None:
        tfm_train.append(norm); tfm_test.append(norm)

    train = datasets.MNIST(root=root, train=True, download=download, transform=T.Compose(tfm_train))
    test  = datasets.MNIST(root=root, train=False, download=download, transform=T.Compose(tfm_test))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **windowsafe_loader(num_workers, pin_memory))
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False, **windowsafe_loader(num_workers, pin_memory))
    return train_loader, test_loader, list(map(str, range(10)))

# -------- ImageFolder (labeled) --------
def get_imagefolder_loaders(base_dir: str, batch_size=128, img_size=224, val_split=0.1,
                            num_workers=0, pin_memory=False, normalize: Optional[dict]=None, in_ch: int=3):
    """Returns DataLoader instances for ImageFolder datasets with train/val/test splits."""
    train_dir = os.path.join(base_dir, 'train')
    val_dir   = os.path.join(base_dir, 'val')
    test_dir  = os.path.join(base_dir, 'test')

    norm = _make_norm_transform(normalize, in_ch)
    def _tf_train():
        xs = [T.Resize((img_size, img_size)), T.RandomHorizontalFlip(), T.ToTensor()]
        if norm is not None: xs.append(norm)
        return T.Compose(xs)
    def _tf_eval():
        xs = [T.Resize((img_size, img_size)), T.ToTensor()]
        if norm is not None: xs.append(norm)
        return T.Compose(xs)

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_ds = CleanImageFolder(train_dir, transform=_tf_train())
        val_ds   = CleanImageFolder(val_dir,   transform=_tf_eval())
    elif os.path.isdir(train_dir) and os.path.isdir(test_dir):
        full = CleanImageFolder(train_dir, transform=_tf_train())
        n_val = max(1, int(len(full)*val_split))
        n_tr  = len(full) - n_val
        train_ds, val_ds = random_split(full, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    else:
        raise ValueError(f"Expected one of:\n  - base_dir/train/, base_dir/val/\n  - base_dir/train/, base_dir/test/ (val split from train)\nGiven base_dir={base_dir}")

    classes = getattr(getattr(train_ds, 'dataset', train_ds), 'classes', None) or []
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **windowsafe_loader(num_workers, pin_memory))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **windowsafe_loader(num_workers, pin_memory))
    return train_loader, val_loader, classes

def get_imagefolder_loaders_testonly(test_dir: str, batch_size=128, img_size=28,
                      num_workers=0, pin_memory=False, as_rgb=False, normalize: Optional[dict]=None, in_ch: int=1):
    """Returns DataLoader instances for ImageFolder datasets (test-only)."""
    tfm_test  = [T.Resize((img_size, img_size))]
    if as_rgb:
        tfm_test .insert(0, T.Grayscale(num_output_channels=3))
        in_ch_eff = 3
    else:
        in_ch_eff = in_ch
    
    xs = [T.Resize((img_size, img_size)), T.ToTensor()]
    norm = _make_norm_transform(normalize, in_ch_eff)
    if norm is not None:
        xs.append(norm)
    tf_eval = T.Compose(xs)
    test_ds = CleanImageFolder(test_dir, transform=tf_eval)
    test_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **windowsafe_loader(num_workers, pin_memory))
    return test_loader, list(map(str, range(10)))

# -------- Unlabeled --------
class UnlabeledImageFolder(Dataset):
    """A dataset for loading unlabeled images from a folder structure."""
    def __init__(self, root, img_size=224, in_ch: int=3, normalize: Optional[dict]=None,
                 extensions=("*.jpg","*.jpeg","*.png","*.bmp")):
        """Initializes the UnlabeledImageFolder dataset.

        Args:
            root: Root directory containing the images.
            img_size: Target image size.
            in_ch: Number of input channels.
            normalize: Normalization dictionary.
            extensions: Tuple of image file extensions to include.
        """
        self.paths = []
        for ext in extensions:
            self.paths += glob.glob(os.path.join(root, "**", ext), recursive=True)
        self.paths = sorted(list(dict.fromkeys(self.paths)))
        self.uids  = list(range(len(self.paths)))
        xs = [T.Resize((img_size, img_size)), T.ToTensor()]
        norm = _make_norm_transform(normalize, in_ch)
        if norm is not None: xs.append(norm)
        self.tfm = T.Compose(xs)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        x = self.tfm(Image.open(p).convert("RGB"))
        uid = self.uids[i]
        return x, uid

def get_unlabeled_loader(base_dir, batch_size=128, img_size=224, num_workers=0, pin_memory=False,
                         shuffle=True, in_ch: int=3, normalize: Optional[dict]=None):
    """Returns a DataLoader for unlabeled images."""
    ds = UnlabeledImageFolder(base_dir, img_size=img_size, in_ch=in_ch, normalize=normalize)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **windowsafe_loader(num_workers, pin_memory))

# -------- Label/Seed Utilities --------
def _resolve_dataset(loader_or_dataset):
    """Resolves a DataLoader or Dataset object to a Dataset."""
    if isinstance(loader_or_dataset, DataLoader):
        return loader_or_dataset.dataset
    if hasattr(loader_or_dataset, "__len__"):
        return loader_or_dataset
    raise ValueError("Unsupported type for loader_or_dataset")

def _labels_of_dataset(ds) -> Optional[Sequence[int]]:
    """Extracts labels from a dataset if available."""
    if hasattr(ds, "targets"):
        y = ds.targets
        if isinstance(y, torch.Tensor):
            y = y.tolist()
        return list(y)
    if hasattr(ds, "samples"):
        return [cls for _, cls in ds.samples]
    if hasattr(ds, "labels"):
        y = ds.labels
        if isinstance(y, torch.Tensor):
            y = y.tolist()
        return list(y)
    return None

def dataset_has_labels(loader_or_dataset) -> bool:
    """Checks if a dataset or DataLoader has associated labels."""
    ds = _resolve_dataset(loader_or_dataset)
    return _labels_of_dataset(ds) is not None

def make_seed_from_labeled(loader_or_dataset, per_class: int, seed: int = 42,
                           batch_size: int = 128, allow_undersample: bool = False, shuffle: bool = True):
    """Creates a seed DataLoader from a labeled dataset, ensuring a certain number of samples per class."""
    ds = _resolve_dataset(loader_or_dataset)
    labels = _labels_of_dataset(ds)
    if labels is None:
        raise ValueError("Dataset has no labels; cannot use --seed-per-class.")

    rng = np.random.default_rng(seed)
    classes = sorted(set(labels))
    idxs = []
    by_class = {c: [] for c in classes}
    for i, yy in enumerate(labels):
        by_class[yy].append(i)

    for c in classes:
        pool = by_class[c]
        if len(pool) < per_class:
            if not allow_undersample:
                raise ValueError(f"Class {c} has only {len(pool)} samples < per_class={per_class}")
            take = pool
        else:
            take = rng.choice(pool, size=per_class, replace=False).tolist()
        idxs.extend(take)

    if shuffle:
        rng.shuffle(idxs)

    subset = Subset(ds, idxs)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

def make_seed_from_pool(loader_or_dataset, size: int, seed: int = 42, batch_size: int = 128, shuffle: bool = True):
    """Creates a seed DataLoader by randomly sampling from a pool of data."""
    ds = _resolve_dataset(loader_or_dataset)
    n = len(ds)
    size = max(1, min(int(size), n))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(n, size=size, replace=False).tolist()
    if shuffle:
        rng.shuffle(idxs)
    subset = Subset(ds, idxs)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

# -------- CIFAR10 --------
class CIFAR10BatchDataset(Dataset):
    """
    A custom dataset for loading CIFAR-10 from batch files.
    """
    def __init__(self, root, train=True, transform=None, download=False):
        """Initializes the CIFAR10BatchDataset.

        Args:
            root: Root directory containing the CIFAR-10 batch files.
            train: If True, loads the training set; otherwise, loads the test set.
            transform: Optional transform to be applied on a sample.
            download: If True, downloads the dataset from the internet and puts it in root directory.
                      (Not implemented for this custom loader).
        """
        self.root = root
        self.transform = transform
        self.train = train
        self.data = []
        self.targets = []

        if download:
            # In a real scenario, you'd implement download logic here.
            # For now, we assume the files are already extracted.
            pass

        if self.train:
            for i in range(1, 6):
                batch_path = os.path.join(root, f"data_batch_{i}")
                with open(batch_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    self.targets.extend(entry['labels'])
        else:
            batch_path = os.path.join(root, "test_batch")
            with open(batch_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1)) # convert to HWC

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

def get_cifar10_loaders(batch_size=128, img_size=32, root='./data/cifar-10-batches-py', download=False,
                        num_workers=0, pin_memory=False, normalize: Optional[dict]=None, in_ch: int=3):
    """Returns DataLoader instances for the CIFAR-10 dataset."""
    tfm_train = [T.Resize((img_size, img_size)), T.RandomHorizontalFlip(), T.ToTensor()]
    tfm_test  = [T.Resize((img_size, img_size)), T.ToTensor()]

    norm = _make_norm_transform(normalize, in_ch)
    if norm is not None:
        tfm_train.append(norm)
        tfm_test.append(norm)

    train_dataset = CIFAR10BatchDataset(root=root, train=True, download=download, transform=T.Compose(tfm_train))
    test_dataset  = CIFAR10BatchDataset(root=root, train=False, download=download, transform=T.Compose(tfm_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **windowsafe_loader(num_workers, pin_memory))
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **windowsafe_loader(num_workers, pin_memory))

    return train_loader, test_loader, list(map(str, range(10))) # CIFAR-10 has 10 classes

# -------- (선택) 라벨 유무에 따라 자동 분기하는 평가용 헬퍼 --------
def get_imagefolder_or_unlabeled(base_dir: str, batch_size=128, img_size=224, normalize: Optional[dict]=None, in_ch: int=3):
    """Returns DataLoader instances for ImageFolder or unlabeled images, automatically detecting dataset type."""
    try:
        tr, ev, classes = get_imagefolder_loaders(base_dir, batch_size, img_size, normalize=normalize, in_ch=in_ch)
        return tr, ev, classes
    except Exception:
        # unlabeled fallback → evaluation set (ev) has no actual labels, so metrics are limited
        unl = get_unlabeled_loader(base_dir, batch_size, img_size, normalize=normalize, in_ch=in_ch)
        return unl, unl, []