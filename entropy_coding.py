import math
import torch
from coinpp.patching import Patcher
from helpers import get_datasets_and_converter


def compute_frequency_distribution(quantized_mods, num_bits, min_count=1):
    """Returns the distribution of modulation values.

    Args:
        quantized_mods (torch.Tensor): Tensor of ints, containing quantized
            modulation values.
        num_bits (int): The number of bits used for quantizing the distribution.
            We cannot safely deduce this from quantized_mods since the maximum
            value might not occur in the data.
        min_count (int): If certain values do not occur in quantized_mods tensor
            add min_count number of counts to this value. This ensures none of
            the values have zero probability which can lead to infinite values
            for the entropy coding calculations.
    """
    # Returns a tensor of shape (num_vals,) containing the number of occurences
    # of each value in the quantized_mods tensor. Note that bincount only works
    # with 1d tensors, so flatten modulations
    counts = torch.bincount(quantized_mods.view(-1))
    # Extends counts tensor if largest values do not occur in quantized_mods
    # tensor
    if len(counts) != 2**num_bits:
        # Check number of distinct values is at most equal to number of bins
        assert (
            len(counts) < 2**num_bits
        ), "quantized_mods values exceed number of bins."
        counts = torch.cat(
            [
                counts,
                torch.zeros(2**num_bits - len(counts), dtype=counts.dtype).to(
                    counts.device
                ),
            ]
        )
        assert len(counts) == 2**num_bits
    # If min_count is not zero, add min_count number of counts to any value that
    # does not occur in tensor
    if min_count:
        counts[counts == 0] += min_count
    # Convert to frequencies
    frequencies = counts.float() / counts.sum()
    return frequencies


def cross_entropy(train_distribution, test_distribution):
    """Calculates the cross entropy between a train (q) and test distribution
    (p), i.e. H(p, q) = - E_p[log q]. This corresponds to the bitrate achieved
    by the model q on the test set (drawn from a distribution p).

    Args:
        train_distribution (torch.Tensor):
        test_distribution (torch.Tensor):

    Notes:
        Returns the cross entropy value in bits *not* nats (i.e. we use log
        base 2, not base e).
    """
    return -(test_distribution * torch.log2(train_distribution)).sum().item()


def get_bpd(bits_per_modulation, num_modulations, num_dims):
    """Calculates the bits per dimension of a model.

    Args:
        bits_per_modulation (float): Number of bits required to store a single
            modulation.
        num_modulations (int): Number of modulations of model.
        num_dims (int): Number of dimensions of *data* (e.g. image, MRI scan,
            etc). For images, we typically choose the number of pixels (rather
            than the number of dimensions which is 3x the number of pixels for
            RGB images) to obtain bits per pixel (bpp).
    """
    # In order to dequantize the data, we are required to store the mean and std
    # values in the quantizer class, each of which are 32 bit floats, so add
    # 2 * 32 bits to the modulation bits. However, this need not be stored for
    # each datapoint/modulations individually, but rather is shared across the
    # entire dataset. We can therefore store this as part of the base network
    # that is shared across datapoints. We therefore do not need to add 2 * 32
    # bits here
    total_bits = bits_per_modulation * num_modulations  # + 2 * 32
    return total_bits / num_dims


def dataset_to_num_dims(dataset_name):
    if dataset_name == "mnist":
        return 28 * 28
    if dataset_name == "cifar10":
        return 32 * 32
    if dataset_name == "era5":
        return 46 * 90


def get_bpd_patched(bits_per_modulation, args):
    """Calculates the bits per dimension of a batched model, where each
    datapoint may have different shapes.

    Args:
        bits_per_modulation (float): Number of bits required to store a single
            modulation.
        args: The arguments obtained from calling load_model.
    """
    modulations_per_patch = args.latent_dim
    patcher = Patcher(args.patch_shape)

    # Create a dataloader to iterate over dataset
    _, test_dataset, _ = get_datasets_and_converter(args)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=1
    )

    # Calculate number of bits per dim for each datapoint. As each datapoint can
    # have a different number of dimensions (e.g. the CLIC dataset has images of
    # different sizes), we must:
    # 1. Iterate over each datapoint individually
    # 2. Check how many patches are required to cover the datapoint
    # 3. Calculate the total number of modulations required to store this
    # datapoint
    # 4. Divide this number by the number of dimensions of the input datapoint
    num_bits_per_dim = []
    for data in test_dataloader:
        # Remove batch dimension and convert to patches
        data = data[0]
        patches, spatial_shape = patcher.patch(data)

        # Patches has shape (num_patches, *)
        num_patches = patches.shape[0]
        # Number of dimensions of signal is product of spatial shape. E.g. for
        # images spatial_shape = (H, W) so num_dims = H * W
        num_dims = math.prod(spatial_shape)
        # We store modulations separately for each patch, so total number of
        # modulations is given by number of patches * modulations per patch
        num_modulations = modulations_per_patch * num_patches
        total_bits = bits_per_modulation * num_modulations
        # Divide by the number of dimensions of the data (which can be different
        # for different datapoints in the dataset)
        num_bits_per_dim.append(total_bits / num_dims)

    return sum(num_bits_per_dim) / len(num_bits_per_dim)


if __name__ == "__main__":
    import argparse
    import wandb
    import wandb_utils

    unit_test = False

    if not unit_test:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--wandb_run_path",
            help="Path of wandb run for trained model.",
            type=str,
            default="nfrc/emi/runs/3vg7g9lh",
        )

        parser.add_argument(
            "--train_mod_dataset",
            help="Name of training modulation dataset.",
            type=str,
            default="modulations_test_3_steps_5_bits_quantized.pt",
        )

        parser.add_argument(
            "--test_mod_dataset",
            help="Name of test modulation dataset.",
            type=str,
            default="modulations_test_3_steps_5_bits_quantized.pt",
        )

        args = parser.parse_args()

        # Load modulations
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_mods = wandb_utils.load_modulations(
            args.wandb_run_path, args.train_mod_dataset, device
        )
        test_mods = wandb_utils.load_modulations(
            args.wandb_run_path, args.test_mod_dataset, device
        )
        # If we are using patching, stack modulation lists into a single
        # modulation tensor
        use_patching = type(train_mods) is list
        if use_patching:
            train_mods = torch.cat(train_mods, dim=0)
            test_mods = torch.cat(test_mods, dim=0)
        # Check modulation ranges
        print(f"Train mods range: {train_mods.min()}, {train_mods.max()}")
        print(f"Test mods range: {test_mods.min()}, {test_mods.max()}")

        # Extract num_bits from file name. Filename has the format
        # ..._<num_bits>_bits_quantized.pt, so use appropriate splitting to
        # extract num_bits
        num_bits = int(args.train_mod_dataset.split("_bits")[0].split("_")[-1])

        # Compute expected number of bits with entropy coding
        train_dist = compute_frequency_distribution(train_mods, num_bits)
        # Note that, unlike the train distribution, for the test distribution
        # we do not need all values to have non zero probability, so set
        # min_count to zero
        test_dist = compute_frequency_distribution(test_mods, num_bits, min_count=0)
        num_bits_entropy_coded = cross_entropy(train_dist, test_dist)

        print(f"Number of bits (without entropy coding): {num_bits}")
        print(f"Expected number of bits: {num_bits_entropy_coded}")
        print(
            f"Entropy coding improvement: {100 * (num_bits - num_bits_entropy_coded)/num_bits:.2f}%"
        )

        # Compute bpd/bpp
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, model_args, _ = wandb_utils.load_model(args.wandb_run_path, device)
        if use_patching:
            bpd = get_bpd_patched(num_bits_entropy_coded, model_args)
        else:
            num_dims = dataset_to_num_dims(model_args.test_dataset)
            bpd = get_bpd(num_bits_entropy_coded, model_args.latent_dim, num_dims)

        print(f"Bits per dimension/pixel: {bpd}")
    else:
        # Test on dummy data
        train_quantized_mods = torch.LongTensor([0, 0, 1, 1, 2, 3])
        test_quantized_mods = torch.LongTensor([0, 1, 1, 2, 2, 3, 3])
        num_bits = 2
        train_dist = compute_frequency_distribution(train_quantized_mods, num_bits)
        test_dist = compute_frequency_distribution(test_quantized_mods, num_bits)
        print(train_dist)
        print(test_dist)
        print(cross_entropy(train_dist, test_dist))
        print(cross_entropy(test_dist, test_dist))

        # Test with missing values
        quantized_mods = torch.LongTensor([0, 0, 0, 0, 2, 3])
        num_bits = 2
        dist = compute_frequency_distribution(quantized_mods, num_bits)
        print(dist)
        dist_no_min_count = compute_frequency_distribution(
            quantized_mods, num_bits, min_count=0
        )
        print(dist_no_min_count)

        # Test with missing values and missing range
        quantized_mods = torch.LongTensor([1, 1, 1, 1, 2, 2])
        num_bits = 2
        dist = compute_frequency_distribution(quantized_mods, num_bits)
        print(dist)
        dist_no_min_count = compute_frequency_distribution(
            quantized_mods, num_bits, min_count=0
        )
        print(dist_no_min_count)
