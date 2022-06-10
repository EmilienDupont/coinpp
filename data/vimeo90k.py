import imageio
import requests
import torch
import torchvision
import zipfile

from pathlib import Path
from typing import Any, Callable, Optional
import os
import shutil


class Vimeo90k(torch.utils.data.Dataset):
    """Vimeo90k triplet dataset. Based on the dataset and pre-processing from CompressAI
    at: https://github.com/InterDigitalInc/CompressAI/issues/105.

    Args:

    """

    urls = [
        "http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip",
    ]

    zip_files = [
        "vimeo_triplet.zip",
    ]

    def __init__(
        self,
        root: Path,
        download: bool = False,
        train: bool = True,
        transform: Optional[Callable] = None,
        verify: bool = False,
    ):
        self.root = root

        self.transform = transform

        if download:
            self.download()
        elif verify and not self._check_exists():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        split = "train" if train else "test"
        self.image_files = list((self.root / split).glob("**/*.png"))

    def __getitem__(self, index: int) -> Any:
        path = self.image_files[index]
        image = imageio.imread(path)

        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_files)

    def _check_exists(self) -> bool:
        return self.root.exists()

    def extract_dataset_split(self, in_dir: str, out_dir: str, list_filename: str):
        with open(list_filename) as f:
            lines = f.read().splitlines()

        os.makedirs(out_dir, exist_ok=True)

        in_dir_path = Path(in_dir)
        out_dir_path = Path(out_dir)

        for subdir in lines:
            if subdir == "":
                continue

            subdir_path = in_dir_path / subdir
            out_prefix = str(out_dir_path / (subdir.replace("/", "_") + "_"))
            in_images = os.listdir(subdir_path)

            for image in in_images:
                src = subdir_path / image
                dst = out_prefix + image
                print(f"{src} -> {dst}", subdir)
                shutil.copy2(src, dst)

    def download(self, chunk_size=1024):
        if self._check_exists():
            print("Vimeo90k files already downloaded ✅")
            return

        self.root.mkdir(parents=True, exist_ok=True)

        print(
            f"Downloading Vimeo90k dataset to {self.root}. This can take a few hours..."
        )

        for url in self.urls:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                filename = url.split("/")[-1]
                with open(self.root / filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)

        for filename in self.zip_files:
            with zipfile.ZipFile(self.root / filename, "r") as z:
                z.extractall(self.root)

        print("Preprocessing the Vimeo90k dataset, this can take a few hours... ")
        self.extract_dataset_split(
            self.root / "vimeo_triplet/sequences",
            self.root / "train",
            self.root / "vimeo_triplet/tri_trainlist.txt",
        )

        self.extract_dataset_split(
            self.root / "vimeo_triplet/sequences",
            self.root / "test",
            self.root / "vimeo_triplet/tri_testlist.txt",
        )

        print(f"Done!")
