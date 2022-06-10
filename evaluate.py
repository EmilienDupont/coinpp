import coinpp.conversion as conversion
import coinpp.losses as losses
import torch


def evaluate_batch(model, converter, modulations, data):
    """Evaluate a batch of modulations.

    Args:
        model (nn.Module):
        converter ():
        modulations (torch.Tensor): Shape (batch_size, num_modulations).
        data (torch.Tensor): Batch of data. Shape (batch_size, channels, *).
    """
    with torch.no_grad():
        coordinates, features = converter.to_coordinates_and_features(data)
        features_recon = model.modulated_forward(coordinates, modulations)
        per_example_mse = losses.batch_mse_fn(features_recon, features)
        # Compute MSE and mean PSNR value across batch
        mse = per_example_mse.mean().item()
        psnr = losses.mse2psnr(per_example_mse).mean().item()
    return mse, psnr


def evaluate_patches(model, converter, patcher, modulations, data, chunk_size=None):
    """Evaluate modulations for a single datapoint split into patches.

    Args:
        model (nn.Module):
        converter ():
        patcher:
        modulations (torch.Tensor): Shape (num_patches, num_modulations).
        data (torch.Tensor): Single datapoint. Shape (channels, *).
        chunk_size (int or None): If not None, evaluates the data in chunks
            instead of evaluating all patches in parallel. This reduces memory
            consumption.
    """
    with torch.no_grad():
        patches, spatial_shape = patcher.patch(data)
        coordinates, features = converter.to_coordinates_and_features(patches)
        # Optionally evaluate model in chunks
        if chunk_size is None:
            # Shape (num_patches, *patch_shape, feature_dim)
            features_recon = model.modulated_forward(coordinates, modulations)
        else:
            # Split coordinates and modulations into batches of size chunk_size
            # and evaluate them in sequence
            # Calculate number of batches of size chunk_size needed to process
            # datapoint
            num_patches = coordinates.shape[0]
            num_batches = num_patches // chunk_size
            last_batch_size = num_patches % chunk_size
            # Iterate over chunks
            features_recon = []
            idx = 0
            for _ in range(num_batches):
                next_idx = idx + chunk_size
                features_recon.append(
                    model.modulated_forward(
                        coordinates[idx:next_idx], modulations[idx:next_idx]
                    )
                )
                idx = next_idx
            # If non zero final batch size, evaluate final piece of data
            if last_batch_size:
                features_recon.append(
                    model.modulated_forward(coordinates[idx:], modulations[idx:])
                )
            # Aggregate all chunks to get tensor of shape
            # (num_patches, *patch_shape, feature_dim)
            # Shape (num_patches, *patch_shape, feature_dim)
            features_recon = torch.cat(features_recon, dim=0)

        # When using patches, we cannot directly calculate MSE and PSNR since
        # creating patches may require us to pad the data. We therefore need to
        # convert the features back to data and unpatch (which takes care of
        # removing the padding) before calculating MSE and PSNR.
        # Shape (num_patches, feature_dim, *patch_shape)
        patch_data = conversion.features2data(features_recon, batched=True)
        # Unpatch data, to obtain shape (feature_dim, *spatial_shape)
        data_recon = patcher.unpatch(patch_data, data.shape[1:])
        # Calculate MSE and PSNR values
        mse = losses.mse_fn(data_recon, data)
        psnr = losses.mse2psnr(mse)
    return mse.item(), psnr.item()


def evaluate_dataset(
    model, converter, patcher, modulations, dataset, batch_size=100, verbose=True
):
    """Evaluate a dataset of modulations. Returns MSE and mean PSNR across
    dataset.

    Args:
        model:
        converter:
        patcher:
        modulations (torch.Tensor or list of torch.Tensor): If tensor, should
            have shape (dataset_size, num_modulations), otherwise should contain
            a list of tensors of shape (num_patches, num_modulations).
        dataset:
        batch_size: Batch size to use when evaluating modulations. Note that when
            patcher is not None, this is overwritten to 1 (as the size of each
            datapoint can be different, we cannot batch).
        verbose:
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size if patcher is None else 1,
    )

    mses = []
    psnrs = []
    idx = 0
    for i, data in enumerate(dataloader):
        if verbose:
            print(f"Batch {i + 1}/{len(dataloader)}")
        # Last batch may be smaller than batch_size, so use data.shape[0]
        next_idx = idx + data.shape[0]
        # Extract modulations corresponding to data batch
        modulations_batch = modulations[idx:next_idx]
        if patcher is None:
            data = data.to(modulations.device)
            mse, psnr = evaluate_batch(model, converter, modulations_batch, data)
        else:
            # When using patches, modulations is a list so modulations_batch
            # will be a list containing one element. Extract this to get a
            # tensor of shape (num_patches, modulations_size)
            modulations_batch = modulations_batch[0]
            # Remove batch dimension from data
            data = data.to(modulations_batch.device)[0]
            mse, psnr = evaluate_patches(
                model,
                converter,
                patcher,
                modulations_batch,
                data,
                chunk_size=batch_size,
            )
        mses.append(mse)
        psnrs.append(psnr)
        idx = next_idx
    return torch.Tensor(mses).mean().item(), torch.Tensor(psnrs).mean().item()


if __name__ == "__main__":
    import argparse
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
        "--modulation_dataset",
        help="Name of modulation dataset to evaluate.",
        type=str,
        default="modulations_test_3_steps.pt",
    )

    parser.add_argument(
        "--batch_size",
        help="Batch size to use when evaluating modulations.",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--verbose", help="Whether to print progress.", type=int, default=1
    )

    args = parser.parse_args()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args, patcher = wandb_utils.load_model(args.wandb_run_path, device)
    # Load dataset
    train_dataset, test_dataset, converter = get_datasets_and_converter(
        model_args, force_no_random_crop=True
    )
    if "train" in args.modulation_dataset:
        dataset = train_dataset
    elif "test" in args.modulation_dataset:
        dataset = test_dataset
    # Load modulations
    modulations = wandb_utils.load_modulations(
        args.wandb_run_path, args.modulation_dataset, device
    )

    # Compute mean MSE and PSNR for entire modulation dataset
    mean_mse, mean_psnr = evaluate_dataset(
        model,
        converter,
        patcher,
        modulations,
        dataset,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    print(f"MSE: {mean_mse}, PSNR: {mean_psnr}")
