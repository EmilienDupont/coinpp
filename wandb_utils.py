import torch
import wandb
from coinpp.patching import Patcher
from helpers import get_model


def load_model(wandb_run_path, device):
    """Load a trained model.

    Args:
        wandb_run_path (string): Path to wandb run, e.g. nfrc/emi/runs/3vg7g9lh.
    """
    # Define local dir based on run id
    run_id = wandb_run_path.split("/")[-1]
    local_dir = f"wandb/{run_id}"
    # Download model from wandb path (and overwrite if it is already there)
    run = wandb.Api().run(wandb_run_path)
    run.file("model.pt").download(root=local_dir, replace=True)
    # Load downloaded model
    if device == torch.device("cpu"):
        model_dict = torch.load(f"{local_dir}/model.pt", map_location=device)
    else:
        model_dict = torch.load(f"{local_dir}/model.pt")
    # Extract args
    args = model_dict["args"]
    args.device = device  # Test device may be different from train device
    # Reconstruct model based on args
    model = get_model(args)
    # Load trained weights into model
    model.load_state_dict(model_dict["state_dict"])
    model = model.to(device)
    # Optionally build patcher
    patcher = None
    if hasattr(args, "patch_shape") and args.patch_shape != [-1]:
        patcher = Patcher(args.patch_shape)
    return model, args, patcher


def load_modulations(wandb_run_path, filename, device):
    """Load a set of modulations.

    Args:
        wandb_run_path (string): Path to wandb run, e.g. 'nfrc/emi/runs/3vg7g9lh'.
        filename (string): Name of modulations file, e.g. 'modulations_test_3_steps.pt'.
    """
    # Define local dir based on run id
    run_id = wandb_run_path.split("/")[-1]
    local_dir = f"wandb/{run_id}"
    # Download modulations from wandb path (and overwrite if it is already there)
    run = wandb.Api().run(wandb_run_path)
    run.file(f"wandb/{run_id}/{filename}").download(root=".", replace=True)
    # Load modulations tensor
    if device == torch.device("cpu"):
        modulations = torch.load(
            f"{local_dir}/{filename}", map_location=torch.device("cpu")
        )
    else:
        modulations = torch.load(f"{local_dir}/{filename}")
    # If modulations is a list, we are using patching. Iterate over every
    # element of list and transfer each tensor to correct device
    if type(modulations) is list:
        modulations = [mods.to(device) for mods in modulations]
    else:
        modulations = modulations.to(device)
    return modulations


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model without patcher
    model, args, patcher = load_model("nfrc/emi/runs/3vg7g9lh", device)
    print(model)
    print(args.device)
    print(patcher)

    # Load model with patcher
    model, args, patcher = load_model("nfrc/emi/runs/2oijh9g0", device)
    print(patcher)
    print(patcher.patch_shape)

    # Load modulations without patcher
    modulations = load_modulations(
        "nfrc/emi/runs/r173ao0o", "modulations_test_3_steps.pt", device
    )
    print(modulations.shape)
    print(modulations.device)

    # Load modulations with patcher
    modulations = load_modulations(
        "nfrc/emi/runs/2oijh9g0", "modulations_test_3_steps.pt", device
    )
    print(len(modulations))
    print(modulations[0].device)
