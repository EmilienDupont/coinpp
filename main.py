import argparse
import helpers
import os
import pprint
import torch
import wandb
from coinpp.patching import Patcher
from coinpp.training import Trainer
from pathlib import Path


def add_arguments(parser):
    """Model, training and wandb arguments. Note that we use integers for boolean
    arguments due to compatibility issues with wandb.
    """
    # Model arguments
    parser.add_argument(
        "--num_layers", help="Number of layers in base network.", type=int, default=5
    )

    parser.add_argument(
        "--dim_hidden",
        help="Dimension of hidden layers of base network.",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--w0", help="w0 parameter from SIREN.", type=float, default=30.0
    )

    parser.add_argument(
        "--modulate_scale",
        help="Whether to modulate scale.",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--modulate_shift",
        help="Whether to modulate shift.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--use_latent",
        help="Whether to use latent vector.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--latent_dim",
        help="Dimension of the latent vector mapped to modulations. If set to -1, will not use latent vector.",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--modulation_net_dim_hidden",
        help="Dimension of hidden layers in modulation network.",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--modulation_net_num_layers",
        help="Number of layers in modulation network. 1 corresponds to a linear layer.",
        type=int,
        default=1,
    )

    # Training arguments
    parser.add_argument(
        "--seed",
        help="Random seed. If set to -1, seed is chosen at random.",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--outer_lr", help="Learning rate for the outer loop.", type=float, default=3e-6
    )

    parser.add_argument(
        "--inner_lr", help="Learning rate for the inner loop.", type=float, default=1e-2
    )

    parser.add_argument(
        "--inner_steps", help="Number of inner loop steps.", type=int, default=3
    )

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--num_epochs", type=int, default=100)

    parser.add_argument(
        "--train_dataset",
        default="mnist",
        choices=(
            "mnist",
            "cifar10",
            "kodak",
            "fastmri",
            "era5",
            "vimeo90k",
            "librispeech",
        ),
    )

    parser.add_argument(
        "--test_dataset",
        default="mnist",
        choices=(
            "mnist",
            "cifar10",
            "kodak",
            "fastmri",
            "era5",
            "vimeo90k",
            "librispeech",
        ),
    )

    parser.add_argument(
        "--num_workers",
        help="Number of workers for dataloader.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradient_checkpointing",
        help="Whether to use gradient checkpointing.",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num_validation_points",
        help="Number of validation data points to use. If -1, will use all available points.",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--validate_every",
        help="Run validation every {validate_every} iterations.",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--validation_inner_steps",
        help="List of inner steps to use for validation.",
        nargs="+",
        type=int,
        default=[3],
    )

    parser.add_argument(
        "--patch_shape",
        help="Shape of patches to use during training. If set to [-1] will not apply patching.",
        nargs="+",
        type=int,
        default=[-1],
    )

    parser.add_argument(
        "--subsample_num_points",
        help="Number of coordinate points to subsample during training. If -1, uses full datapoint/patch.",
        type=int,
        default=-1,
    )

    # Wandb arguments
    parser.add_argument(
        "--use_wandb",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="debug",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--wandb_job_type",
        help="Wandb job type. This is useful for grouping runs together.",
        type=str,
        default=None,
    )


def main(args):
    if args.use_wandb:
        # Initialize wandb experiment
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            job_type=args.wandb_job_type,
            config=args,
        )

        # Save ENV variables
        with (Path(wandb.run.dir) / "env.txt").open("wt") as f:
            pprint.pprint(dict(os.environ), f)

        # Define path where model will be saved
        model_path = Path(wandb.run.dir) / "model.pt"
    else:
        model_path = ""

    # Optionally set random seed
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Build datasets, converters and model
    train_dataset, test_dataset, converter = helpers.get_datasets_and_converter(args)
    model = helpers.get_model(args)

    print(model)
    print(args)

    # Optionally build patcher
    if args.patch_shape != [-1]:
        patcher = Patcher(args.patch_shape)
    else:
        patcher = None

    # Optionally save model
    if args.use_wandb:
        torch.save({"args": args, "state_dict": model.state_dict()}, model_path)
        wandb.save(str(model_path.absolute()), base_path=wandb.run.dir, policy="live")

    # Initialize trainer and start training
    trainer = Trainer(
        func_rep=model,
        converter=converter,
        args=args,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        patcher=patcher,
        model_path=model_path,
    )

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}:")
        trainer.train_epoch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    add_arguments(parser)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
