import coinpp.conversion as conversion
import coinpp.models as models
import data.audio as audio
import data.era5 as era5
import data.fastmri as fastmri
import data.image as image
import data.vimeo90k as vimeo90k
import torchvision
import yaml
from pathlib import Path


def get_dataset_root(dataset_name: str):
    """Returns path to data based on dataset_paths.yaml file."""
    with open(r"data/dataset_paths.yaml") as f:
        dataset_paths = yaml.safe_load(f)

    return Path(dataset_paths[dataset_name])


def dataset_name_to_dims(dataset_name):
    """Returns appropriate dim_in and dim_out for dataset."""
    if dataset_name == "mnist":
        dim_in, dim_out = 2, 1
    if dataset_name in ("cifar10", "kodak", "vimeo90k"):
        dim_in, dim_out = 2, 3
    if dataset_name == "fastmri":
        dim_in, dim_out = 3, 1
    if dataset_name == "era5":
        dim_in = 3
        dim_out = 1
    if dataset_name == "librispeech":
        dim_in = 1
        dim_out = 1
    return dim_in, dim_out


def get_datasets_and_converter(args, force_no_random_crop=False):
    """Returns train and test datasets as well as appropriate data converters.

    Args:
        args: Arguments parsed from input.
        force_no_random_crop (bool): If True, forces datasets to not use random
            crops (which is the default for the training set when using
            patching). This is useful after the model is trained when we store
            modulations.
    """
    # Extract input and output dimensions of function rep
    dim_in, dim_out = dataset_name_to_dims(args.train_dataset)

    # When using patching, perform random crops equal to patch size on training
    # dataset
    use_patching = hasattr(args, "patch_shape") and args.patch_shape != [-1]
    if use_patching:
        if dim_in == 2:
            random_crop = torchvision.transforms.RandomCrop(args.patch_shape)

    if "mnist" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter("image")

        if args.train_dataset == "mnist":
            train_dataset = image.MNIST(
                root=get_dataset_root("mnist"),
                train=True,
                transform=random_crop if use_patching else None,
            )
        if args.test_dataset == "mnist":
            test_dataset = image.MNIST(root=get_dataset_root("mnist"), train=False)

    if "cifar10" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter("image")

        if args.train_dataset == "cifar10":
            transforms = [torchvision.transforms.ToTensor()]
            if use_patching and not force_no_random_crop:
                transforms.append(random_crop)

            train_dataset = image.CIFAR10(
                root=get_dataset_root("cifar10"),
                train=True,
                transform=torchvision.transforms.Compose(transforms),
            )
        if args.test_dataset == "cifar10":
            test_dataset = image.CIFAR10(
                root=get_dataset_root("cifar10"),
                train=False,
                transform=torchvision.transforms.ToTensor(),
            )

    if "kodak" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter("image")

        if args.train_dataset == "kodak":
            transforms = [torchvision.transforms.ToTensor()]
            if use_patching and not force_no_random_crop:
                transforms.append(random_crop)

            train_dataset = image.Kodak(
                root=get_dataset_root("kodak"),
                download=True,
                transform=torchvision.transforms.Compose(transforms),
            )

        if args.test_dataset == "kodak":
            test_dataset = image.Kodak(
                root=get_dataset_root("kodak"),
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )

    if "fastmri" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter("mri")

        if args.train_dataset == "fastmri":
            train_dataset = fastmri.FastMRI(
                root=get_dataset_root("fastmri"),
                split="train",
                challenge="multicoil",
                patch_shape=args.patch_shape
                if (use_patching and not force_no_random_crop)
                else -1,
            )

        if args.test_dataset == "fastmri":
            test_dataset = fastmri.FastMRI(
                root=get_dataset_root("fastmri"),
                split="val",
                challenge="multicoil",
            )

    if "era5" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter("era5")

        if args.train_dataset == "era5":
            train_dataset = era5.ERA5(root=get_dataset_root("era5"), split="train")

        if args.test_dataset == "era5":
            test_dataset = era5.ERA5(root=get_dataset_root("era5"), split="test")

    if "vimeo90k" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter("image")

        if args.train_dataset == "vimeo90k":
            transforms = [torchvision.transforms.ToTensor()]
            if use_patching and not force_no_random_crop:
                transforms.append(random_crop)

            # As Vimeo90k dataset is large, do not download by default. If you wish to
            # download this dataset, manually set download=True.
            train_dataset = vimeo90k.Vimeo90k(
                root=get_dataset_root("vimeo90k"),
                train=True,
                download=False,
                transform=torchvision.transforms.Compose(transforms),
            )

        if args.test_dataset == "vimeo90k":
            test_dataset = vimeo90k.Vimeo90k(
                root=get_dataset_root("vimeo90k"),
                train=False,
                download=False,
                transform=torchvision.transforms.ToTensor(),
            )

    if "librispeech" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter("audio")

        # We use first 3 seconds of each audio sample
        if args.train_dataset == "librispeech":
            train_dataset = audio.LIBRISPEECH(
                root=get_dataset_root("librispeech"),
                url="train-clean-100",
                patch_shape=args.patch_shape[0]
                if (use_patching and not force_no_random_crop)
                else -1,
                num_secs=3,
                download=True,
            )
        if args.test_dataset == "librispeech":
            test_dataset = audio.LIBRISPEECH(
                root=get_dataset_root("librispeech"),
                url="test-clean",
                num_secs=3,
                download=True,
            )

    return train_dataset, test_dataset, converter


def get_model(args):
    dim_in, dim_out = dataset_name_to_dims(args.train_dataset)
    return models.ModulatedSiren(
        dim_in=dim_in,
        dim_hidden=args.dim_hidden,
        dim_out=dim_out,
        num_layers=args.num_layers,
        w0=args.w0,
        w0_initial=args.w0,
        modulate_scale=args.modulate_scale,
        modulate_shift=args.modulate_shift,
        use_latent=args.use_latent,
        latent_dim=args.latent_dim,
        modulation_net_dim_hidden=args.modulation_net_dim_hidden,
        modulation_net_num_layers=args.modulation_net_num_layers,
    ).to(args.device)
