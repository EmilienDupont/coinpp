import coinpp.metalearning as metalearning
import time
import torch
from helpers import get_datasets_and_converter


def compute_modulations(
    model,
    args,
    patcher,
    inner_steps,
    use_train_dataset=True,
    batch_size=64,
    timing="off",
):
    """Given a trained model and a dataset, fit modulations to entire dataset.

    Args:
        model:
        args:
        patcher:
        inner_steps:
        use_train_dataset:
        batch_size: Batch size to use when computing modulations. Note that when
            patcher is not None, this is overwritten to 1 (as the size of each
            datapoint can be different, we cannot batch).
        timing: One of "off", "per_example", "per_dataset".

    Returns:
        A tensor of shape (num_datapoints, modulations_size) containing all
        modulations if patcher is None, otherwise a list of length
        num_datapoints containing modulation tensors of shape
        (num_patches, modulations_size), where num_patches can be different for
        different list entries.
    """
    # If we are measuring encoding time per example, must use batch_size of 1
    if timing == "per_example":
        batch_size = 1

    # Initialize times if timing
    if timing != "off":
        times = []

    # Move stored modulations to CPU so they don't take up GPU RAM
    cpu_device = torch.device("cpu")

    # Extract dataset and create dataloader
    train_dataset, test_dataset, converter = get_datasets_and_converter(
        args, force_no_random_crop=True
    )

    if use_train_dataset:
        dataset = train_dataset
    else:
        dataset = test_dataset

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size if patcher is None else 1,
    )

    all_modulations = []

    for i, data in enumerate(dataloader):
        print(f"Batch {i + 1}/{len(dataloader)}")

        if timing != "off":
            start = time.time()

        data = data.to(args.device)
        if patcher:
            # If using patching, data will have a batch size of 1.
            # Remove batch dimension and instead convert data into
            # patches, with patch dimension acting as batch size
            patches, spatial_shape = patcher.patch(data[0])
            coordinates, features = converter.to_coordinates_and_features(patches)
            # Use batch size argument as chunk size
            outputs = metalearning.outer_step_chunked(
                model,
                coordinates,
                features,
                inner_steps=inner_steps,
                inner_lr=args.inner_lr,
                chunk_size=batch_size,
                gradient_checkpointing=args.gradient_checkpointing,
            )
        else:
            coordinates, features = converter.to_coordinates_and_features(data)
            outputs = metalearning.outer_step(
                model,
                coordinates,
                features,
                inner_steps=inner_steps,
                inner_lr=args.inner_lr,
                is_train=False,
                return_reconstructions=False,
                gradient_checkpointing=args.gradient_checkpointing,
            )

        if timing != "off":
            end = time.time()
            times.append(end - start)

        # outputs['modulations'] has shape (batch_size, num_modulations)
        all_modulations.append(outputs["modulations"].detach().to(cpu_device))

    if patcher is None:
        # Stack to create a tensor of shape (num_datapoints, num_modulations)
        # Note that ehen using patches, number of modulations might be different
        # for different entries, so we do not stack tensor and instead keep it
        # as a list
        all_modulations = torch.cat(all_modulations, dim=0)

    assert len(all_modulations) == len(dataset)

    if timing == "per_example":
        mean_encoding_time = sum(times) / len(times)
        print(f"Per example encoding time: {mean_encoding_time}s")

    if timing == "per_dataset":
        total_encoding_time = sum(times)
        print(f"Total encoding time: {total_encoding_time}s")

    return all_modulations


if __name__ == "__main__":
    import argparse
    import wandb
    from wandb_utils import load_model

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb_run_path",
        help="Path of wandb run for trained model.",
        type=str,
        default="nfrc/emi/runs/3vg7g9lh",
    )

    parser.add_argument(
        "--inner_steps",
        help="Number of inner steps to take when fitting modulations.",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--use_train_dataset",
        help="Whether to fit modulations on train dataset. If False, fits on test set.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch_size",
        help="Batch size to use when fitting modulations.",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--store",
        help="Whether to store modulations.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--timing",
        help="Whether to time how long it takes to encode data. If per_example,"
        " measures time it takes to encode a single example (i.e. forcing "
        "a batch size of 1), if per_dataset measures time to encode "
        "entire dataset (with potentially large batch size).",
        default="off",
        choices=("off", "per_example", "per_dataset"),
    )

    args = parser.parse_args()

    # Compute modulations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args, patcher = load_model(args.wandb_run_path, device)
    modulations = compute_modulations(
        model,
        model_args,
        patcher,
        args.inner_steps,
        use_train_dataset=args.use_train_dataset,
        batch_size=args.batch_size,
        timing=args.timing,
    )

    # Save modulations locally and then upload them to the wandb run
    if args.store:
        run_id = args.wandb_run_path.split("/")[-1]
        local_dir = f"wandb/{run_id}"
        dataset_split = "train" if args.use_train_dataset else "test"
        filename = f"modulations_{dataset_split}_{args.inner_steps}_steps.pt"
        torch.save(modulations, f"{local_dir}/{filename}")
        run = wandb.Api().run(args.wandb_run_path)
        run.upload_file(f"{local_dir}/{filename}")
