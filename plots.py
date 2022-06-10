import glob
import imageio
import json5 as json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import re
import torch
import torchvision
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# Ensure script can be executed even if cartopy is not available
try:
    import cartopy.crs as ccrs
except:
    cartopy_available = False
else:
    cartopy_available = True


ours = "COIN++"

# Setup colormap for residuals plot and num_bits ablations
viridis = cm.get_cmap("viridis", 100)

# Ensure consistent coloring across plots
name_to_color = {
    ours: mcolors.TABLEAU_COLORS["tab:blue"],
    "COIN": mcolors.TABLEAU_COLORS["tab:purple"],
    "BMS": mcolors.TABLEAU_COLORS["tab:brown"],
    "MBT": mcolors.TABLEAU_COLORS["tab:orange"],
    "CST": mcolors.TABLEAU_COLORS["tab:red"],
    "JPEG": mcolors.TABLEAU_COLORS["tab:green"],
    "JPEG2000": mcolors.TABLEAU_COLORS["tab:olive"],
    "BPG": mcolors.TABLEAU_COLORS["tab:pink"],
    "VTM": mcolors.TABLEAU_COLORS["tab:gray"],
    "MP3": mcolors.TABLEAU_COLORS["tab:orange"],
    "3 steps": "#82bfe9",
    "10 steps": mcolors.TABLEAU_COLORS["tab:blue"],
    "50 steps": "#092132",
    "Base": "#77b41f",
    "+ quantization": "#b41f77",
    "+ entropy coding": mcolors.TABLEAU_COLORS["tab:blue"],
    "3 bits": viridis(0.0),
    "4 bits": viridis(0.2),
    "5 bits": viridis(0.4),
    "6 bits": viridis(0.6),
    "7 bits": viridis(0.8),
    "8 bits": viridis(1.0),
    "128": viridis(0.0),
    "256": viridis(0.2),
    "384": viridis(0.4),
    "512": viridis(0.6),
    "768": viridis(0.8),
    "1024": viridis(1.0),
}

name_to_path = {
    ours: "coinpp.json",
    "COIN": "coin.json",
    "BMS": "compressai-bmshj2018-hyperprior.json",
    "MBT": "compressai-mbt2018.json",
    "CST": "compressai-cheng2020-anchor.json",
    "JPEG": "jpeg.json",
    "JPEG2000": "jpeg2000.json",
    "BPG": "bpg_444_x265_ycbcr.json",
    "VTM": "vtm.json",
    "MP3": "mp3.json",
    "Base": "ablations/coinpp_base.json",
    "+ quantization": "ablations/coinpp_with_quantization.json",
    "+ entropy coding": "ablations/coinpp_full.json",
    "COIN curve": "ablations/coin_encoding_curve.json",
    "COIN++ curve": "ablations/coinpp_encoding_curve.json",
    "COIN quantization": "ablations/coin_quantization.json",
    "COIN++ quantization": "ablations/coinpp_quantization.json",
}

dataset_to_dir = {
    "Kodak": "results/kodak/",
    "CLIC": "results/clic/",
    "FastMRI": "results/fastmri/",
    "ERA5": "results/era5/",
    "CIFAR10": "results/cifar10/",
    "MNIST": "results/mnist/",
    "LIBRISPEECH": "results/librispeech/",
}


def parse_json_file(filepath, metric="psnr", inner_steps=None, num_bits=None):
    """Parses a json result file.

    Args:
        filepath (string): Path to results json file.
        metric (string): Metric to use for plot.
        inner_steps (int or None): Only required when loading ablation files.
            Number of inner steps used to evaluate model.
        num_bits (int or None): Only required when loading ablation files.
            Number of bits used to evaluate model.

    Notes:
        Based on https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/utils/plot/__main__.py
    """
    filepath = Path(filepath)
    name = filepath.name.split(".")[0]
    with filepath.open("r") as f:
        data = json.load(f)

    if "results" not in data:
        raise ValueError(f"Invalid file {filepath}")

    results = data["results"]

    if inner_steps is not None:
        results = results[f"{inner_steps} steps"]

    if num_bits is not None:
        results = results[f"{num_bits} bits"]

    if metric == "ms-ssim":
        # Convert to db
        values = np.array(results[metric])
        results[metric] = -10 * np.log10(1 - values)

    return {
        "name": data.get("name", name),
        "xs": results["bpp"],
        "ys": results[metric],
    }


def rate_distortion(
    scatters,
    title=None,
    ylabel="PSNR [dB]",
    output_file=None,
    limits=None,
    show=False,
    figsize=None,
    bpp_lim_zero=True,
):
    """Creates a rate distortion plot based on scatters.

    Args:
        scatters (list of dicts): List of data to plot for each model.
        title (string):
        ylabel (string):
        output_file (string): If not None, save plot at output_file.
        limits (tuple of ints):
        show (bool): If True shows plot.
        figsize (tuple of ints):
        bpp_lim_zero (bool): If True, sets x-axis (i.e. bpp) lower limit to
            zero.

    Notes:
        Based on https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/utils/plot/__main__.py
    """
    if figsize is None:
        figsize = (7, 4)
    fig, ax = plt.subplots(figsize=figsize)
    for sc in scatters:
        if sc["name"] in [ours]:
            linewidth = 2.5
            markersize = 10
        elif sc["name"] in [
            "COIN",
            "BMS",
            "MBT",
            "CST",
            "JPEG",
            "JPEG2000",
            "BPG",
            "VTM",
            "MP3",
        ]:
            linewidth = 1
            markersize = 6
        else:
            linewidth = 1.5
            markersize = 8

        if sc["name"] in ["JPEG", "JPEG2000", "BPG", "VTM", "MP3"]:
            pattern = ".--"  # Non learned algorithms
        else:
            pattern = ".-"  # Learned algorithms

        ax.plot(
            sc["xs"],
            sc["ys"],
            pattern,
            label=sc["name"],
            c=name_to_color[sc["name"]],
            linewidth=linewidth,
            markersize=markersize,
        )

    ax.set_xlabel("Bit-rate [bpp]")
    ax.set_ylabel(ylabel)
    ax.grid()
    if bpp_lim_zero:
        ax.set_xlim(0.0)
    if limits is not None:
        ax.axis(limits)
    ax.legend(loc="lower right")

    if title:
        ax.title.set_text(title)

    if show:
        plt.show()

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.clf()
        plt.close()


def plot_rate_distortion(
    dataset="Kodak",
    models=["COIN++", "COIN", "BMS", "MBT", "CST", "JPEG", "JPEG2000", "BPG", "VTM"],
    output_file=None,
    limits=None,
):
    """Creates rate distortion plot based on all results json files.

    Args:
        dataset (string): Dataset for which to create rate distortion plot. One
            of 'CLIC', 'Kodak', 'CIFAR10', 'FastMRI' and 'ERA5'.
        models (list of string): List of models to include in plot.
        output_file (string): Path to save image.
        limits (tuple of float): Limits of plot.
    """
    # Build filepaths based on dataset and baselines
    filepaths = [dataset_to_dir[dataset] + name_to_path[model] for model in models]
    # Read data
    scatters = []
    for f in filepaths:
        rv = parse_json_file(f, "psnr")
        scatters.append(rv)
    # Create plot
    rate_distortion(scatters, output_file=output_file, limits=limits)


def plot_ablation_rate_distortion(
    plot_type,
    dataset="CIFAR10",
    inner_steps=None,
    num_bits=None,
    quantize=True,
    entropy_code=True,
    output_file=None,
    limits=None,
):
    """Creates rate distortion plots for ablations on CIFAR10.

    Args:
        plot_type (string): One of 'quantization_num_bits', 'num_inner_steps',
            'entropy_coding_quantization'.
        dataset (string): Dataset for which to create plot. One of 'CLIC',
            'Kodak', 'CIFAR10', 'FastMRI' and 'ERA5'.
        inner_steps (int or None):
        num_bits (int or None):
        quantize (bool):
        entropy_code (bool):
        output_file (string): Path to save image.
        limits (tuple of float): Limits of plot.
    """
    if entropy_code:
        assert (
            quantize
        ), f"quantize argument must be True when entropy_code argument is True"

    if plot_type == "quantization_num_bits":
        assert (
            inner_steps is not None
        ), f"Inner steps must be specified for {plot_type} ablation plot."

        if entropy_code:
            filepath = dataset_to_dir[dataset] + name_to_path["+ entropy coding"]
        else:
            filepath = dataset_to_dir[dataset] + name_to_path["+ quantization"]

        # Read data
        scatters = []
        # Quantization is measured for 3, 4, ..., 8 bits for CIFAR10
        if dataset == "CIFAR10":
            num_bits_range = list(range(3, 9))
        # and 5, 6, 7, 8 bits for ERA5
        if dataset == "ERA5":
            num_bits_range = list(range(5, 9))
        # and 5, 6 bits for Kodak
        if dataset == "Kodak":
            num_bits_range = list(range(5, 7))
        # and 5, 6 bits for FastMRI
        if dataset == "FastMRI":
            num_bits_range = list(range(5, 7))
        for num_bits_ in num_bits_range:
            rv = parse_json_file(
                filepath, "psnr", num_bits=num_bits_, inner_steps=inner_steps
            )
            rv["name"] = f"{num_bits_} bits"
            scatters.append(rv)

    if plot_type == "num_inner_steps":
        if quantize:
            assert (
                num_bits is not None
            ), f"Number of bits must be specified for {plot_type} ablation plot if quantize is True."
        else:
            # If we are not quantizing, num_bits must be None
            num_bits = None

        if entropy_code:
            filepath = dataset_to_dir[dataset] + name_to_path["+ entropy coding"]
        elif quantize:
            filepath = dataset_to_dir[dataset] + name_to_path["+ quantization"]
        else:
            filepath = dataset_to_dir[dataset] + name_to_path["Base"]

        # Read data
        scatters = []
        # Performance is measured for 3, 10, 50 inner steps bits
        for inner_steps_ in (3, 10, 50):
            rv = parse_json_file(
                filepath, "psnr", num_bits=num_bits, inner_steps=inner_steps_
            )
            rv["name"] = f"{inner_steps_} steps"
            scatters.append(rv)

    if plot_type == "entropy_coding_quantization":
        assert (
            inner_steps is not None
        ), f"Inner steps must be specified for {plot_type} ablation plot."
        assert (
            num_bits is not None
        ), f"Inner steps must be specified for {plot_type} ablation plot."

        filepaths = [
            dataset_to_dir[dataset] + name_to_path["Base"],
            dataset_to_dir[dataset] + name_to_path["+ quantization"],
            dataset_to_dir[dataset] + name_to_path["+ entropy coding"],
        ]

        # Read data
        scatters = []
        for f in filepaths:
            if f.endswith(name_to_path["Base"]):
                # No quantization in base model
                rv = parse_json_file(f, "psnr", num_bits=None, inner_steps=inner_steps)
            else:
                rv = parse_json_file(
                    f, "psnr", num_bits=num_bits, inner_steps=inner_steps
                )
            scatters.append(rv)

    # Create plot
    rate_distortion(scatters, output_file=output_file, limits=limits)


def plot_encoding_time(
    dataset="CIFAR10",
    models=[
        "BPG",
        "COIN++ 3 steps latent dim 384",
        "COIN++ 10 steps latent dim 384",
        "COIN",
    ],
    output_file=None,
    show=False,
):
    """Plots histogram of encoding time for various codecs.

    Args:
        dataset (string): Dataset for which to create plot. One of 'CLIC',
            'Kodak', 'CIFAR10', 'FastMRI' and 'ERA5'.
        models (list of string): List of models to include in plot.
        output_file (string): If not None, save plot at output_file.
        show (bool): If True shows plot.
    """
    # Open encoding time json file
    filepath = Path(f"{dataset_to_dir[dataset]}/ablations/encoding_time.json")

    with filepath.open("r") as f:
        data = json.load(f)

    # Extract model names and encoding times
    model_names = []
    enc_times = []
    for model_name in models:
        assert (
            model_name in data["results"]["per_example"]
        ), f"{model_name} not available for {dataset}"
        enc_times.append(data["results"]["per_example"][model_name])
        # Update model names for plot
        if model_name.startswith("COIN++ 3 steps"):
            model_name = "COIN++\n(3 steps)"
        if model_name.startswith("COIN++ 10 steps"):
            model_name = "COIN++\n(10 steps)"
        model_names.append(model_name)

    # Create log scale barplot
    plt.grid(zorder=0, which="both", axis="y", alpha=0.2)  # Ensure grid is at the back
    barplot = plt.bar(model_names, enc_times, log=True, zorder=10)
    for i in range(len(model_names)):
        if model_names[i].startswith("COIN++"):
            color = name_to_color["COIN++"]
        else:
            color = name_to_color[model_names[i]]
        barplot[i].set_color(color)
    plt.ylabel("Encoding time [seconds]")

    # Add encoding time as text above each bar
    # Set appropriate limits so we can fit text
    plt.ylim(0.4 * min(enc_times), 3.0 * max(enc_times))
    for i, rect in enumerate(barplot):
        plt.text(
            rect.get_x() + rect.get_width() / 2.0,
            1.2 * rect.get_height(),
            f"{enc_times[i]:.3f}s",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight=400,
        )

    fig = plt.gcf()
    fig.set_size_inches(4, 4)

    if show:
        plt.show()

    if output_file:
        plt.savefig(output_file, format="png", dpi=300, bbox_inches="tight")
        plt.clf()
        plt.close()


def plot_encoding_curve(
    dataset="CIFAR10",
    models=["COIN++", "COIN"],
    max_iters=None,
    log_scale=False,
    output_file=None,
    show=False,
):
    """Plots encoding curves (PSNR vs number of iterations).

    Args:
        dataset (string): Dataset for which to create plot. One of 'CLIC',
            'Kodak', 'CIFAR10', 'FastMRI' and 'ERA5'.
        models (list of string): List of models to include in plot. Must be
            'COIN' and/or 'COIN++'.
        max_iters (int): If not None, fix x limit of plot at max_iters.
        log_scale (bool): If True, uses log scale for x axis.
        output_file (string): If not None, save plot at output_file.
        show (bool): If True shows plot.
    """
    num_iters = 0
    for model in models:
        # Read data
        filepath = Path(f'{dataset_to_dir[dataset]}/{name_to_path[model + " curve"]}')
        with filepath.open("r") as f:
            data = json.load(f)

        # Plot mean and standard deviation of psnr at every step
        mean = np.array(data["results"]["psnr_mean"])
        std = np.array(data["results"]["psnr_std"])
        if max_iters is not None:
            mean = mean[:max_iters]
            std = std[:max_iters]
        # Psnrs are measured after 1st iteration, so start at 1
        iterations = np.array(range(len(mean))) + 1
        plt.plot(iterations, mean, c=name_to_color[model], label=model)
        plt.fill_between(
            iterations,
            mean - std,
            mean + std,
            facecolor=name_to_color[model],
            alpha=0.5,
        )
        # If number of iterations in current plot is larger than previous
        # num_iters value, then update
        if len(mean) > num_iters:
            num_iters = len(mean)

    plt.xlabel("Encoding time [iterations]")
    plt.ylabel("PSNR [dB]")
    plt.legend(loc="lower right")

    plt.xlim(1, max_iters if max_iters is not None else num_iters)
    plt.grid()

    # Ensure x-axis only use integer ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if log_scale:
        plt.xscale("log")

    if show:
        plt.show()

    if output_file:
        plt.savefig(output_file, format="png", dpi=300, bbox_inches="tight")
        plt.clf()
        plt.close()


def plot_quantization_curve(
    dataset="CIFAR10",
    models=["COIN++", "COIN"],
    plot_latent_dim_ablation=False,
    output_file=None,
    show=False,
):
    """Plots quantization curves (number of bits vs PSNR).

    Args:
        dataset (string): Dataset for which to create plot. Currently only
            available for CIFAR10.
        models (list of string): List of models to include in plot. Must be
            'COIN' and/or 'COIN++'.
        plot_latent_dim_ablation (bool): If True, ignores models argument and
            instead plots an ablation of quantization on COIN++ for different
            latent dims.
        output_file (string): If not None, save plot at output_file.
        show (bool): If True shows plot.
    """
    if plot_latent_dim_ablation:
        # Override models argument with different latent dimensions in ablation
        models = ["128", "256", "384", "512", "768", "1024"]

    for model in models:
        # Read data
        if plot_latent_dim_ablation:
            filepath = Path(
                f'{dataset_to_dir[dataset]}/{name_to_path["COIN++ quantization"]}'
            )
        else:
            filepath = Path(
                f'{dataset_to_dir[dataset]}/{name_to_path[model + " quantization"]}'
            )
        with filepath.open("r") as f:
            data = json.load(f)

        # Extract appropriate results
        if plot_latent_dim_ablation:
            results = data["results"][model]
        else:
            if model == "COIN++":
                results = data["results"]["384"]
            else:
                results = data["results"]

        # Extract full precision PSNR (i.e. 32 bits)
        fp_psnr = results["32"]

        # Calculate PSNR drop
        psnr_drop = []
        for num_bits in range(1, 17):
            psnr_drop.append(results[str(num_bits)] - fp_psnr)

        # Measure quantization from 1 to 16 bits
        all_num_bits = list(range(1, 17))
        plt.plot(all_num_bits, psnr_drop, ".-", c=name_to_color[model], label=model)

    plt.xlabel("Number of bits")
    plt.ylabel("PSNR drop [dB]")
    plt.legend(loc="lower right")

    plt.xlim(1, 16)
    plt.ylim(-20, 0.5)
    plt.grid()

    fig = plt.gcf()
    fig.set_size_inches(6, 3.5)

    if show:
        plt.show()

    if output_file:
        plt.savefig(output_file, format="png", dpi=300, bbox_inches="tight")
        plt.clf()
        plt.close()


def plot_qualitative_comparison(
    models=["COIN++", "BPG"],
    dataset="CIFAR10",
    max_residual=0.15,
    upscale=1,
    padding_fraction=0.1,
    output_file=None,
    show=False,
):
    """Plots qualitative comparisons between models, by showing original and
    compressed data side by side, as well as a residual plot of the differences.

    Args:
        models (list of string): List of models to include in plot.
        dataset (string): Dataset for which to create plot.
        max_residual (float): Value between 0 and 1 to use for maximum residual
            on color scale. Usually set to a low value so residuals are clearer
            on plot.
        upscale (int): If not 1, upscales the size of image by a factor of
            upscale.
        padding_fraction (float): Padding to use as a fraction of image height.
        output_file (string): If not None, save plot at output_file.
        show (bool): If True shows plot.
    """
    # Dictionary mapping model name to image path roots
    model_to_path = {"COIN++": "coinpp", "BPG": "bpg"}

    data_dir = f"{dataset_to_dir[dataset]}/ablations/qualitative"

    to_tensor = torchvision.transforms.ToTensor()
    # Initialize list of all images to be iterated over
    all_imgs = []
    # Extract paths to original images (or tensors)
    if dataset in ("ERA5", "FastMRI"):
        original_paths = glob.glob(f"{data_dir}/original_*.pt")
    else:
        original_paths = glob.glob(f"{data_dir}/original_*.png")
    # Ensure consistent image ordering
    original_paths = alphanumeric_sort(original_paths)

    # Iterate over original images and find corresponding reconstructions
    for original_path in original_paths:
        # Add original image
        if dataset == "ERA5":
            # Tensor of shape (1, num_lats, num_lons)
            original_temperatures = torch.load(original_path)
            # Change file extension from .pt to .png
            original_path_img = original_path[:-2] + "png"
            # Render temperature data on globe
            globe_plot(original_temperatures, output_file=original_path_img, dpi=200)
            # Load rendered image
            original_img = imageio.imread(original_path_img)
            all_imgs.append(to_tensor(original_img))
        elif dataset == "FastMRI":
            # Tensor of shape (1, 16, 320, 320)
            original_mri = torch.load(original_path)
            # Take central slice of MRI scan
            original_mri = original_mri[:, 8]
            # Copy data across channel dim to obtain image of shape (3, 320, 320)
            original_mri = original_mri.repeat(3, 1, 1)
            all_imgs.append(original_mri)
        else:
            original_img = imageio.imread(original_path)
            all_imgs.append(to_tensor(original_img))

        # Extract image index from file name (see original_paths)
        img_idx = original_path.split("/")[-1].split(".")[0].split("_")[-1]
        for model in models:
            # Load model reconstructed image
            if dataset == "ERA5":
                rec_path = f"{data_dir}/{model_to_path[model]}_{img_idx}.pt"
                rec_temperatures = torch.load(rec_path)
                # Render image of globe
                img_path = f"{data_dir}/{model_to_path[model]}_{img_idx}.png"
                globe_plot(rec_temperatures, output_file=img_path, dpi=200)
                # Add reconstructed image
                model_img = imageio.imread(img_path)
                all_imgs.append(to_tensor(model_img))
            elif dataset == "FastMRI":
                rec_path = f"{data_dir}/{model_to_path[model]}_{img_idx}.pt"
                rec_mri = torch.load(rec_path)
                # Take central slice of MRI scan
                rec_mri = rec_mri[:, 8]
                # Copy data across channel dim to obtain image of shape (3, 320, 320)
                rec_mri = rec_mri.repeat(3, 1, 1)
                all_imgs.append(rec_mri)
            else:
                img_path = f"{data_dir}/{model_to_path[model]}_{img_idx}.png"
                # Add reconstructed image
                model_img = imageio.imread(img_path)
                all_imgs.append(to_tensor(model_img))

            if dataset == "ERA5":
                residual_temperatures = torch.abs(
                    original_temperatures - rec_temperatures
                )
                residual_path = (
                    f"{data_dir}/{model_to_path[model]}_{img_idx}_residual.png"
                )
                # Render residual plot using viridis cmap
                globe_plot(
                    residual_temperatures,
                    output_file=residual_path,
                    vmin=0,
                    vmax=max_residual,
                    cmap="viridis",
                    dpi=200,
                )
                residual_img = imageio.imread(residual_path)
            elif dataset == "FastMRI":
                # Shape (320, 320) since we take mean across channels
                residual_mri = torch.abs(original_mri - rec_mri).mean(dim=0)
                residual_img = viridis(residual_mri.numpy() / max_residual)[:, :, :3]
            else:
                # Calculate residual image (take mean resiudal across RGB channels)
                # Note that viridis returns 4 channels (RGB and alpha), so remove
                # last channel
                residual_img = viridis(
                    np.abs(original_img / 255.0 - model_img / 255.0).mean(axis=-1)
                    / max_residual
                )[:, :, :3]
            all_imgs.append(to_tensor(residual_img))

    # Stack all images to yield a tensor of shape (num_imgs, channels, height, width)
    all_imgs = torch.stack(all_imgs)
    # Optionally increase size of images
    if upscale != 1:
        all_imgs = magnify(all_imgs, upscale)
    # Set image grid parameters
    # Each row contains original image, plus a reconstruction and residual image
    # for each model
    num_imgs_per_row = 1 + 2 * len(models)
    # Padding as a fraction of image height
    padding = int(padding_fraction * all_imgs.shape[-2])

    if show:
        img_grid = torchvision.utils.make_grid(
            all_imgs, nrow=num_imgs_per_row, padding=padding, pad_value=1
        )
        plt.imshow(img_grid)

    if output_file:
        torchvision.utils.save_image(
            all_imgs, output_file, nrow=num_imgs_per_row, padding=padding, pad_value=1
        )


def plot_qualitative_quantization(
    dataset="CIFAR10",
    num_bits=(1, 2, 3, 4, 5),
    upscale=1,
    padding_fraction=0.1,
    output_file=None,
    show=False,
):
    """Plots qualitative comparisons between different quantization, by showing
    original image, reconstructed image and image at various quantization
    bitwidths.

    Args:
        dataset (string): Dataset for which to create plot.
        num_bits (tuple of int): Tuple of bitwidths to include in plot.
        upscale (int): If not 1, upscales the size of image by a factor of
            upscale.
        padding_fraction (float): Padding to use as a fraction of image height.
        output_file (string): If not None, save plot at output_file.
        show (bool): If True shows plot.
    """
    data_dir = f"{dataset_to_dir[dataset]}/ablations/qualitative/quantization"

    to_tensor = torchvision.transforms.ToTensor()
    # Initialize list of all images to be iterated over
    all_imgs = []

    # Extract paths to original images
    original_paths = glob.glob(f"{data_dir}/original_*.png")
    # Ensure consistent ordering
    original_paths = alphanumeric_sort(original_paths)
    num_imgs = len(original_paths)
    # Extract paths to reconstructed images
    reconstructed_paths = glob.glob(f"{data_dir}/reconstruction_*_full.png")
    reconstructed_paths = alphanumeric_sort(reconstructed_paths)
    # Iterate over every bit width and store paths
    paths = {
        "original": original_paths,
        "reconstruction": reconstructed_paths,
    }
    ordered_keys = ["original", "reconstruction"]
    for bits in num_bits:
        quant_paths = glob.glob(f"{data_dir}/reconstruction_*_{bits}_bits.png")
        quant_paths = alphanumeric_sort(quant_paths)
        paths[f"{bits} bits"] = quant_paths
        ordered_keys.append(f"{bits} bits")

    # Load all images
    for key in ordered_keys:
        for path in paths[key]:
            # Add images
            img = imageio.imread(path)
            all_imgs.append(to_tensor(img))

    # Stack all images to yield a tensor of shape (num_imgs, channels, height, width)
    all_imgs = torch.stack(all_imgs)
    # Optionally increase size of images
    if upscale != 1:
        all_imgs = magnify(all_imgs, upscale)
    # Set image grid parameters
    # Padding as a fraction of image height
    padding = int(padding_fraction * all_imgs.shape[-2])

    if show:
        img_grid = torchvision.utils.make_grid(
            all_imgs, nrow=num_imgs, padding=padding, pad_value=1
        )
        plt.imshow(img_grid)

    if output_file:
        torchvision.utils.save_image(
            all_imgs, output_file, nrow=num_imgs, padding=padding, pad_value=1
        )


def plot_meta_learning_curves(dataset="CIFAR10", output_file=None, show=False):
    """Plots meta-learning curves."""
    filepath = f"{dataset_to_dir[dataset]}/ablations/learning_curves.json"
    filepath = Path(filepath)

    with filepath.open("r") as f:
        results = json.load(f)

    # Extract the steps from the results dict
    steps = results.pop("steps")

    if dataset == "LIBRISPEECH":
        # For the librispeech experiments, we used different latent dimensions and
        # patches, so manually set order of curves to plot. Format is
        # "latent dim - patch"
        latent_dims = ["128 - 1600", "128 - 800", "128 - 400", "128 - 200", "256 - 200"]
    else:
        # The remaining items in the dict are latent dims as strings, so convert
        # these to ints
        latent_dims = [int(key) for key in results.keys()]
        # Sort latent dims
        latent_dims = sorted(latent_dims)
    # Compute colors for each latent dim
    colors = [viridis(i / (len(latent_dims) - 1)) for i in range(len(latent_dims))]

    # Iterate over each latent dim and plot meta learning curve
    for i, latent_dim in enumerate(latent_dims):
        psnrs = results[str(latent_dim)]
        plt.plot(steps[: len(psnrs)], psnrs, label=latent_dim, c=colors[i], linewidth=1)

    # Fix limits and add legends and axes
    plt.xlim(0, steps[-1])
    plt.xlabel("Iterations")
    plt.ylabel("PSNR [dB]")
    plt.legend(loc="lower right")
    plt.grid()

    fig = plt.gcf()
    fig.set_size_inches(6, 3.5)

    if dataset == "LIBRISPEECH":
        # Reduce number of ticks for librispeech for clarity
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))

    if show:
        plt.show()

    if output_file:
        plt.savefig(output_file, format="png", dpi=300, bbox_inches="tight")
        plt.clf()
        plt.close()


def alphanumeric_sort(iterable):
    """Sort iterable in alphanumeric order humans would expect. Taken from
    https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python"""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(iterable, key=alphanum_key)


def magnify(img, factor=1):
    """Magnifies an image or a batch of images.

    Args:
        img (torch.Tensor): Shape (batch_size, channels, height, width) or
            (channels, height, width).
    """
    return torch.repeat_interleave(
        torch.repeat_interleave(img, factor, dim=-1), factor, dim=-2
    )


def globe_plot(
    temperature,
    view=(100.0, 0.0),
    flat=False,
    vmin=0.05,
    vmax=0.95,
    cmap="plasma",
    dpi=300,
    output_file=None,
    show=False,
):
    """Plots temperature on a globe.

    Args:
        temperature (array-like): Array of shape (1, num_lats, num_lons) or
            (num_lats, num_lons) containing temperatures.
        view (tuple of floats): Longitude and latitude values for the view of
            the globe.
        flat (bool): If True, creates flat map plot, otherwise plots globe.
        output_file (string): If not None, save plot at output_file.
        show (bool): If True shows plot.

    Notes:
        Taken from https://github.com/EmilienDupont/neural-function-distributions/blob/main/viz/plots_globe.py
    """
    if not cartopy_available:
        raise RuntimeError(
            "The cartopy library is required for globe plots, "
            "but was not found. Please install it to use globe "
            "plots."
        )

    # Set map projection
    if flat:
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        ax = plt.axes(projection=ccrs.Orthographic(*view))

    # Add coastlines
    ax.coastlines()

    # Remove channel dimension
    if temperature.ndim == 3:
        temperature = temperature[0]

    # Create latitude and longitude vectors
    num_lats, num_lons = temperature.shape
    # Uniformly spaced latitudes and longitudes corresponding to ERA5 grids
    latitude = np.linspace(90.0, -90.0, num_lats)
    longitude = np.linspace(0.0, 360.0 - (360.0 / num_lons), num_lons)

    # Plot climate data
    mesh = ax.pcolormesh(
        longitude, latitude, temperature, transform=ccrs.PlateCarree(), cmap=cmap
    )

    if show:
        plt.show()

    if output_file:
        plt.savefig(output_file, format="png", dpi=dpi, bbox_inches="tight")
        plt.clf()
        plt.close()


if __name__ == "__main__":
    import os

    make_plot_rate_distortion = True
    make_plot_encoding_curve = False
    make_plot_encoding_time = False
    make_plot_quantization_curve = False
    make_plot_num_bits_ablation = False
    make_plot_num_inner_steps_ablation = False
    make_plot_quantization_entropy_coding_ablation = False
    make_plot_qualitative_comparison = False
    make_plot_qualitative_quantization = False
    make_plot_learning_curve = True

    if not os.path.exists("figures"):
        os.mkdir("figures")

    if make_plot_rate_distortion:
        plot_rate_distortion(
            "CIFAR10",
            models=["COIN++", "COIN", "BPG", "JPEG2000", "JPEG", "BMS", "CST"],
            output_file="figures/rate_distortion_cifar10.png",
        )
        plot_rate_distortion(
            "ERA5",
            models=["COIN++", "BPG", "JPEG2000", "JPEG"],
            output_file="figures/rate_distortion_era5.png",
        )
        plot_rate_distortion(
            "Kodak",
            models=[
                "COIN++",
                "COIN",
                "BPG",
                "JPEG2000",
                "JPEG",
                "BMS",
                "MBT",
                "CST",
                "VTM",
            ],
            output_file="figures/rate_distortion_kodak.png",
            limits=(0, 1, 22, 38),
        )
        plot_rate_distortion(
            "FastMRI",
            models=["COIN++", "BPG", "JPEG2000", "JPEG"],
            output_file="figures/rate_distortion_fastmri.png",
        )
        plot_rate_distortion(
            "LIBRISPEECH",
            models=["COIN++", "MP3"],
            output_file="figures/rate_distortion_librispeech.png",
        )

    if make_plot_encoding_curve:
        plot_encoding_curve(
            output_file="figures/encoding_curves.png",
        )
        plot_encoding_curve(
            output_file="figures/encoding_curves_log_scale.png", log_scale=True
        )
        plot_encoding_curve(
            output_file="figures/encoding_curves_zoom.png", max_iters=10
        )

    if make_plot_encoding_time:
        plot_encoding_time(output_file="figures/encoding_time.png")

    if make_plot_quantization_curve:
        plot_quantization_curve(output_file="figures/quantization_curves.png")
        plot_quantization_curve(
            output_file="figures/quantization_curves_coinpp_ablation.png",
            plot_latent_dim_ablation=True,
        )

    if make_plot_num_bits_ablation:
        if not os.path.exists("figures/ablation_num_bits"):
            os.mkdir("figures/ablation_num_bits")

        # Ablation for CIFAR10
        for inner_steps in (3, 10, 50):
            plot_ablation_rate_distortion(
                plot_type="quantization_num_bits",
                inner_steps=inner_steps,
                entropy_code=False,
                output_file=f"figures/ablation_num_bits/{inner_steps}_steps.png",
            )
            plot_ablation_rate_distortion(
                plot_type="quantization_num_bits",
                inner_steps=inner_steps,
                entropy_code=True,
                output_file=f"figures/ablation_num_bits/{inner_steps}_steps_entropy_coding.png",
            )
        # Ablation for ERA5
        for inner_steps in (3,):
            plot_ablation_rate_distortion(
                plot_type="quantization_num_bits",
                dataset="ERA5",
                inner_steps=inner_steps,
                entropy_code=False,
                output_file=f"figures/ablation_num_bits/era5/{inner_steps}_steps.png",
            )
            plot_ablation_rate_distortion(
                plot_type="quantization_num_bits",
                dataset="ERA5",
                inner_steps=inner_steps,
                entropy_code=True,
                output_file=f"figures/ablation_num_bits/era5/{inner_steps}_steps_entropy_coding.png",
            )

        # Ablation for Kodak
        for inner_steps in (3, 10):
            plot_ablation_rate_distortion(
                plot_type="quantization_num_bits",
                dataset="Kodak",
                inner_steps=inner_steps,
                entropy_code=False,
                output_file=f"figures/ablation_num_bits/kodak/{inner_steps}_steps.png",
            )
            plot_ablation_rate_distortion(
                plot_type="quantization_num_bits",
                dataset="Kodak",
                inner_steps=inner_steps,
                entropy_code=True,
                output_file=f"figures/ablation_num_bits/kodak/{inner_steps}_steps_entropy_coding.png",
            )

        # Ablation for FastMRI
        for inner_steps in (10,):
            plot_ablation_rate_distortion(
                plot_type="quantization_num_bits",
                dataset="FastMRI",
                inner_steps=inner_steps,
                entropy_code=False,
                output_file=f"figures/ablation_num_bits/fastmri/{inner_steps}_steps.png",
            )
            plot_ablation_rate_distortion(
                plot_type="quantization_num_bits",
                dataset="FastMRI",
                inner_steps=inner_steps,
                entropy_code=True,
                output_file=f"figures/ablation_num_bits/fastmri/{inner_steps}_steps_entropy_coding.png",
            )

    if make_plot_num_inner_steps_ablation:
        if not os.path.exists("figures/ablation_num_inner_steps"):
            os.mkdir("figures/ablation_num_inner_steps")

        # Plot without quantization nor entropy coding
        plot_ablation_rate_distortion(
            plot_type="num_inner_steps",
            quantize=False,
            entropy_code=False,
            output_file=f"figures/ablation_num_inner_steps/base.png",
        )
        # Plot with quantization and entropy coding
        for num_bits in list(range(3, 9)):
            plot_ablation_rate_distortion(
                plot_type="num_inner_steps",
                num_bits=num_bits,
                quantize=True,
                entropy_code=False,
                output_file=f"figures/ablation_num_inner_steps/{num_bits}_bits.png",
            )
            plot_ablation_rate_distortion(
                plot_type="num_inner_steps",
                num_bits=num_bits,
                quantize=True,
                entropy_code=True,
                output_file=f"figures/ablation_num_inner_steps/{num_bits}_bits_entropy_coding.png",
            )

    if make_plot_quantization_entropy_coding_ablation:
        if not os.path.exists("figures/ablation_entropy_coding_quantization"):
            os.mkdir("figures/ablation_entropy_coding_quantization")

        for inner_steps in (3, 10, 50):
            for num_bits in list(range(3, 9)):
                plot_ablation_rate_distortion(
                    plot_type="entropy_coding_quantization",
                    num_bits=num_bits,
                    inner_steps=inner_steps,
                    output_file=f"figures/ablation_entropy_coding_quantization/{inner_steps}_steps_{num_bits}_bits.png",
                )

    if make_plot_qualitative_comparison:
        plot_qualitative_comparison(
            output_file="figures/qualitative_comparison_cifar10.png", upscale=4
        )
        plot_qualitative_comparison(
            output_file="figures/qualitative_comparison_era5.png",
            dataset="ERA5",
            max_residual=0.02,
            models=["COIN++"],
        )
        plot_qualitative_comparison(
            output_file="figures/qualitative_comparison_fastmri.png",
            models=["COIN++"],
            dataset="FastMRI",
            max_residual=0.25,
        )
        plot_qualitative_comparison(
            output_file="figures/qualitative_comparison_kodak.png",
            models=["COIN++"],
            dataset="Kodak",
            max_residual=0.5,
        )

    if make_plot_qualitative_quantization:
        plot_qualitative_quantization(
            output_file="figures/qualitative_quantization_cifar10.png",
            dataset="CIFAR10",
            upscale=4,
        )
        plot_qualitative_quantization(
            output_file="figures/qualitative_quantization_mnist.png",
            dataset="MNIST",
            upscale=4,
        )

    if make_plot_learning_curve:
        plot_meta_learning_curves(
            output_file="figures/meta_learning_curves_cifar10.png", dataset="CIFAR10"
        )
        plot_meta_learning_curves(
            output_file="figures/meta_learning_curves_fastmri.png", dataset="FastMRI"
        )
        plot_meta_learning_curves(
            output_file="figures/meta_learning_curves_kodak.png", dataset="Kodak"
        )
        plot_meta_learning_curves(
            output_file="figures/meta_learning_curves_era5.png", dataset="ERA5"
        )
        plot_meta_learning_curves(
            output_file="figures/meta_learning_curves_librispeech.png",
            dataset="LIBRISPEECH",
        )
