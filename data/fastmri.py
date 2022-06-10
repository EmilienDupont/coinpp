import h5py
import random
import torch
from pathlib import Path
from typing import Callable, List, Optional, Union


class FastMRI(torch.utils.data.Dataset):

    SINGLECOIL_MIN_VAL = 4.4726e-09
    SINGLECOIL_MAX_VAL = 0.0027
    MULTICOIL_MIN_VAL = 1.0703868156269891e-06
    MULTICOIL_MAX_VAL = 0.0007881390047259629

    def __init__(
        self,
        root: Union[str, Path],
        challenge: str,
        split: str,
        normalize: bool = True,
        machine_type: str = "AXT2",
        num_slices: int = 16,
        patch_shape: Union[int, List[int]] = -1,
        transform: Optional[Callable] = None,
    ):
        """
        Dataset of 3D MRI scans.

        Args:
            root (Union[str, Path]): Path to the dataset root
            challenge (str): "singlecoil" or "multicoil"
            split (str, optional): "train", "val", or "test"
            normalize (bool, optional): Whether to normalize data to lie in [0, 1].
                Defaults to True.
            machine_type (str): If not None, machine type to use. Otherwise uses all
                machine types.
            num_slices (int): If not None, filter for volumes with the specified number
                of slices.
            patch_shape (Union[int, List[int]], optional): If not -1, perform random
                crops of shape patch_shape. Defaults to -1.
            transform (Optional[Callable], optional): [description]. Defaults to None.
        """
        if challenge not in ["singlecoil", "multicoil"] or split not in [
            "train",
            "val",
            "test",
        ]:
            raise ValueError

        root = Path(root) / f"{challenge}_{split}"
        files = sorted(list(Path(root).glob("*.h5")))

        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

        if not root.exists or len(files) == 0:
            raise FileNotFoundError

        self.normalize = normalize
        self.machine_type = machine_type
        self.num_slices = num_slices
        self.patch_shape = patch_shape
        self.random_crop = patch_shape != -1
        self.transform = transform

        if challenge == "singlecoil":
            self.min_val = FastMRI.SINGLECOIL_MIN_VAL
            self.max_val = FastMRI.SINGLECOIL_MAX_VAL
        else:
            self.min_val = FastMRI.MULTICOIL_MIN_VAL
            self.max_val = FastMRI.MULTICOIL_MAX_VAL

        # Optionally filter for machine type
        if machine_type is not None:
            valid_files = []
            for path in files:
                if get_machine_type(path) == machine_type:
                    valid_files.append(path)
            self.files = valid_files
        else:
            self.files = files

        # Optionally filter for volumes with a specific number of slices
        if num_slices is not None:
            self.files = list(filter(self.num_slices_equal_to, self.files))

    def num_slices_equal_to(self, file):
        with h5py.File(file, "r") as f:
            mri = torch.from_numpy(f[self.recons_key][()])
            return mri.shape[0] == self.num_slices

    def __getitem__(self, idx):

        with h5py.File(self.files[idx], "r") as f:
            mri = torch.from_numpy(f[self.recons_key][()])

        # Shape ({1,} depth, height, width)
        if mri.ndim == 3:
            # Ensure volume has a channel dimension, i.e. (1, depth, height, width)
            mri = mri.unsqueeze(0)

        # Normalize data to lie in [0, 1]
        if self.normalize:
            mri = (mri - self.min_val) / (self.max_val - self.min_val)
            mri = torch.clamp(mri, 0.0, 1.0)

        if self.transform:
            mri = self.transform(mri)

        if self.random_crop:
            mri = random_crop3d(mri, self.patch_shape)

        return mri

    def __len__(self):
        return len(self.files)


def random_crop3d(data, patch_shape):
    if not (
        0 < patch_shape[0] <= data.shape[-3]
        and 0 < patch_shape[1] <= data.shape[-2]
        and 0 < patch_shape[2] <= data.shape[-1]
    ):
        print(data.shape)
        print(patch_shape)
        raise ValueError("Invalid shapes.")
    depth_from = random.randint(0, data.shape[-3] - patch_shape[0])
    height_from = random.randint(0, data.shape[-2] - patch_shape[1])
    width_from = random.randint(0, data.shape[-1] - patch_shape[2])
    return data[
        ...,
        depth_from : depth_from + patch_shape[0],
        height_from : height_from + patch_shape[1],
        width_from : width_from + patch_shape[2],
    ]


def get_machine_type(path):
    """Returns machine type from path to fastMRI file."""
    # Get filename
    filename = str(path).split("/")[-1]
    # Remove 'file_brain_' string which is at the beginning of every filename
    # Then extract machine type (first word after file_brain_)
    return filename.replace("file_brain_", "").split("_")[0]
