import torch
import coinpp.conversion as conversion


def reconstruct(data, modulations, model, converter, patcher):
    """Reconstructs a single data point.

    Args:
        data: A single datapoint. E.g. a single image. Shape (channels, *spatial_shape).
        modulations: A single set of modulations of shape (1, latent_dim) or
            (num_patches, latent_dim) if using patching.
        model:
        converter:
        patcher:
    """
    with torch.no_grad():
        if patcher is None:
            coordinates, features = converter.to_coordinates_and_features(data)
            features_recon = model.modulated_forward(coordinates, modulations)
            data_recon = conversion.features2data(features_recon, batched=False)
        else:
            patches, spatial_shape = patcher.patch(data)
            coordinates, features = converter.to_coordinates_and_features(patches)
            # Shape (num_patches, *patch_shape, feature_dim)
            features_recon = model.modulated_forward(coordinates, modulations)
            # When using patches, we cannot directly calculate MSE and PSNR since
            # creating patches may require us to pad the data. We therefore need to
            # convert the features back to data and unpatch (which takes care of
            # removing the padding) before calculating MSE and PSNR.
            # Shape (num_patches, feature_dim, *patch_shape)
            patch_data = conversion.features2data(features_recon, batched=True)
            # Unpatch data, to obtain shape (feature_dim, *spatial_shape)
            data_recon = patcher.unpatch(patch_data, data.shape[1:])
    return data_recon


if __name__ == "__main__":
    import argparse
    import os
    import wandb
    import wandb_utils
    from helpers import get_datasets_and_converter
    from torchvision.utils import save_image

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb_run_path",
        help="Path of wandb run for trained model.",
        type=str,
        default="nfrc/emi/runs/3vg7g9lh",
    )

    parser.add_argument(
        "--save_dir",
        help="Directory where data and their reconstructions will be saved.",
        type=str,
        default="./",
    )

    parser.add_argument(
        "--modulation_dataset",
        help="Name of modulation dataset to use to generate reconstructions.",
        type=str,
        default="modulations_test_10_steps_6_bits_dequantized.pt",
    )

    parser.add_argument(
        "--data_indices",
        help="Indices of points in dataset for which original and reconstructions will be saved.",
        nargs="+",
        type=int,
        default=[0],
    )

    args = parser.parse_args()

    # Load modulations
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    modulations = wandb_utils.load_modulations(
        args.wandb_run_path, args.modulation_dataset, device
    )

    # Load model
    model, model_args, patcher = wandb_utils.load_model(args.wandb_run_path, device)
    # Load dataset
    train_dataset, test_dataset, converter = get_datasets_and_converter(
        model_args, force_no_random_crop=True
    )
    # Check if modulations were created from train or test set
    if "train" in args.modulation_dataset:
        dataset = train_dataset
    elif "test" in args.modulation_dataset:
        dataset = test_dataset

    # Create directory to save reconstructions if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for idx in args.data_indices:
        data = test_dataset[idx].to(device)
        if patcher is None:
            # If no patching, extract modulations of shape (1, latent_dim)
            mods = modulations[idx : idx + 1]
        else:
            # modulations is a list of tensors of shape (num_patches, latent_dim)
            # for each data point. Therefore extract single tensor of shape
            # (num_patches, latent_dim)
            mods = modulations[idx]
        data_recon = reconstruct(data, mods, model, converter, patcher)
        # Save original data and reconstruction
        if converter.data_type == "image":
            save_image(data, os.path.join(args.save_dir, f"original_{idx}.png"))
            save_image(
                data_recon, os.path.join(args.save_dir, f"reconstruction_{idx}.png")
            )
        elif converter.data_type in ("mri", "era5"):
            torch.save(data, os.path.join(args.save_dir, f"original_{idx}.pt"))
            torch.save(
                data_recon, os.path.join(args.save_dir, f"reconstruction_{idx}.pt")
            )
