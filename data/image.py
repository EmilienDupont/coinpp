import imageio
import requests
import torch
import torchvision
import zipfile

from pathlib import Path
from typing import Any, Callable, Optional


class CIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR10 dataset without labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index])
        else:
            return self.data[index]


class MNIST(torchvision.datasets.MNIST):
    """MNIST dataset without labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Add channel dimension, convert to float and normalize to [0, 1]
        datapoint = self.data[index].unsqueeze(0).float() / 255.0
        if self.transform:
            return self.transform(datapoint)
        else:
            return datapoint


class Kodak(torch.utils.data.Dataset):
    """Kodak dataset."""

    base_url = "http://r0k.us/graphics/kodak/kodak/"
    num_images = 24
    width = 768
    height = 512
    resolution_hw = (height, width)

    def __init__(
        self,
        root: Path = Path.cwd() / "kodak-dataset",
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.root = root

        self.transform = transform

        if download:
            self.download()

        self.data = tuple(
            imageio.imread(self.root / f"kodim{idx + 1:02}.png")
            for idx in range(self.num_images)
        )

    def _check_exists(self) -> bool:
        # This can be obviously be improved for instance by comparing checksums.
        return (
            self.root.exists() and len(list(self.root.glob("*.png"))) == self.num_images
        )

    def download(self):
        if self._check_exists():
            return

        self.root.mkdir(parents=True, exist_ok=True)

        print(f"Downloading Kodak dataset to {self.root}...")

        for idx in range(self.num_images):
            filename = f"kodim{idx + 1:02}.png"
            with open(self.root / filename, "wb") as f:
                f.write(
                    requests.get(
                        f"http://r0k.us/graphics/kodak/kodak/{filename}"
                    ).content
                )

        print("Done!")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __repr__(self) -> str:
        head = "Dataset Kodak"
        body = []
        body.append(f"Number of images: {self.num_images}")
        body.append(f"Root location: {self.root}")
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)
