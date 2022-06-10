import glob
import numpy as np
import torch


# Statistics for the era5_temp2m_16x_train dataset (in Kelvin)
T_MIN = 202.66
T_MAX = 320.93
T_MEAN = 277.77
T_STD = 21.48


class ERA5(torch.utils.data.Dataset):
    """ERA5 temperature dataset.

    Args:
        root (string or PosixPath): Path to directory where data is stored.
        split (string): Which split to use from train/val/test.
        transform (torchvision.Transform): Optional transform to apply to data.
        normalize (bool): Whether to normalize data to lie in [0, 1]. Defaults to True.
    """

    def __init__(self, root, split, transform=None, normalize=True):
        if split not in ["train", "val", "test"]:
            raise ValueError("Invalid value for split argument")

        self.root = root
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.filepaths = glob.glob(str(root) + f"_{split}/*.npz")
        self.filepaths.sort()  # Ensure consistent ordering of paths

    def __getitem__(self, index):
        # Dictionary containing latitude, longitude and temperature
        data = np.load(self.filepaths[index])
        temperature = data["temperature"]  # Shape (num_lats, num_lons)
        # Optionally normalize data
        if self.normalize:
            temperature = (temperature - T_MIN) / (T_MAX - T_MIN)
        # Convert to tensor and add channel dimension (1, num_lats, num_lons)
        temperature = torch.Tensor(temperature).unsqueeze(0)
        # Perform optional transform
        if self.transform:
            temperature = self.transform(temperature)
        return temperature

    def __len__(self):
        return len(self.filepaths)
