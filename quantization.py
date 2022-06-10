import torch


class Quantizer:
    """Class for quantizing modulations.

    Args:
        mean (float): Mean of modulations.
        std (float): Standard deviation of modulations.
        std_range (float): Number of standard deviations defining quantization
            range. All values lying outside (-std_range, std_range) after
            normalization will be clipped to this range.
    """

    def __init__(self, mean, std, std_range=3.0):
        self.mean = mean
        self.std = std
        self.std_range = std_range

    def quantize(self, modulations, num_bits):
        """Uniformly quantize modulations to a given number of bits.

        Args:
            modulations (torch.Tensor):
            num_bits (int): Number of bits at which to quantize. This
                corresponds to uniformly quantizing into 2 ** num_bits bins.
        """
        # Normalize modulations
        norm_mods = (modulations - self.mean) / self.std
        # Clip modulations to lie in quantization range
        norm_mods = torch.clamp(norm_mods, -self.std_range, self.std_range)
        # Map modulations from [-std_range, std_range] to [0, 1]
        norm_mods = norm_mods / (2 * self.std_range) + 0.5
        # Compute number of bins
        num_bins = 2**num_bits
        # Quantize modulations. After multiplying by (num_bins - 1) this will
        # yield values in [0, num_bins - 1]. Rounding will then yield values in
        # {0, 1, ..., num_bins - 1}, i.e. num_bins different values
        quantized_mods = torch.round(norm_mods * (num_bins - 1))
        # Dequantize modulations
        dequantized_norm_mods = quantized_mods / (num_bins - 1)
        dequantized_norm_mods = (dequantized_norm_mods - 0.5) * 2 * self.std_range
        dequantized_mods = dequantized_norm_mods * self.std + self.mean
        return quantized_mods.int(), dequantized_mods


if __name__ == "__main__":
    import argparse
    import evaluate
    import wandb
    import wandb_utils
    from helpers import get_datasets_and_converter

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb_run_path",
        help="Path of wandb run for trained model.",
        type=str,
        default="nfrc/emi/runs/3vg7g9lh",
    )

    parser.add_argument(
        "--train_mod_dataset",
        help="Name of modulation dataset to create quantizer.",
        type=str,
        default="modulations_test_3_steps.pt",
    )

    parser.add_argument(
        "--test_mod_dataset",
        help="Name of modulation dataset to quantize.",
        type=str,
        default="modulations_test_3_steps.pt",
    )

    parser.add_argument(
        "--num_bits",
        help="List of number of bits at which to quantize modulations.",
        nargs="+",
        type=int,
        default=[5],
    )

    parser.add_argument(
        "--store",
        help="Whether to store quantized modulations.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--evaluate",
        help="Whether to evaluate PSNR of quantized modulations.",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch_size",
        help="Batch size to use when evaluating modulations.",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--std_range",
        help="Quantization range (in number of standard devs).",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()

    # Load modulations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_modulations = wandb_utils.load_modulations(
        args.wandb_run_path, args.train_mod_dataset, device
    )
    test_modulations = wandb_utils.load_modulations(
        args.wandb_run_path, args.test_mod_dataset, device
    )

    # Optionally load model and dataset if evaluating
    if args.evaluate:
        model, model_args, patcher = wandb_utils.load_model(args.wandb_run_path, device)
        # Load dataset
        train_dataset, test_dataset, converter = get_datasets_and_converter(
            model_args, force_no_random_crop=True
        )
        # Check if test modulations were created from train or test set
        if "train" in args.test_mod_dataset:
            dataset = train_dataset
        elif "test" in args.test_mod_dataset:
            dataset = test_dataset

    # If modulations is a list, we are using patching
    use_patching = type(train_modulations) is list

    # Define quantizer
    if use_patching:
        # When using patching, modulations contains a list of tensors of
        # shape (num_patches, num_modulations) where each num_patches might
        # be different for different entries. We therefore stack all entries
        # before calculating the mean and std
        stacked_modulations = torch.cat(train_modulations, dim=0)
        mean = stacked_modulations.mean().item()
        std = stacked_modulations.std().item()
    else:
        mean = train_modulations.mean().item()
        std = train_modulations.std().item()
    quantizer = Quantizer(mean, std, std_range=args.std_range)

    # Extract information to for saving quantized modulations on wandb
    run_id = args.wandb_run_path.split("/")[-1]
    local_dir = f"wandb/{run_id}"
    run = wandb.Api().run(args.wandb_run_path)
    modulations_base = args.test_mod_dataset.split(".")[0]

    # Quantize at various bitwidths and save modulations
    for num_bits in args.num_bits:
        if use_patching:
            # With patching, iterate over list of modulations and quantize
            # each of them individually
            quantized, dequantized = [], []
            for modulation in test_modulations:
                quantized_single, dequantized_single = quantizer.quantize(
                    modulation, num_bits=num_bits
                )
                quantized.append(quantized_single)
                dequantized.append(dequantized_single)
        else:
            quantized, dequantized = quantizer.quantize(
                test_modulations, num_bits=num_bits
            )
        if args.store:
            # Save modulations locally and then upload them to the wandb run
            filename_quant = f"{modulations_base}_{num_bits}_bits_quantized.pt"
            filename_dequant = f"{modulations_base}_{num_bits}_bits_dequantized.pt"
            torch.save(quantized, f"{local_dir}/{filename_quant}")
            torch.save(dequantized, f"{local_dir}/{filename_dequant}")
            run.upload_file(f"{local_dir}/{filename_quant}")
            run.upload_file(f"{local_dir}/{filename_dequant}")

        if args.evaluate:
            # Compute mean MSE and PSNR for entire modulation dataset
            mean_mse, mean_psnr = evaluate.evaluate_dataset(
                model,
                converter,
                patcher,
                dequantized,
                dataset,
                batch_size=args.batch_size,
                verbose=False,
            )
            print(f"{num_bits} bits: {mean_psnr} dB")
