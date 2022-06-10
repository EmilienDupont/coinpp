import coinpp.losses as losses
import torch
import torch.utils.checkpoint as cp


def inner_loop(
    func_rep,
    modulations,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
):
    """Performs inner loop, i.e. fits modulations such that the function
    representation can match the target features.

    Args:
        func_rep (models.ModulatedSiren):
        modulations (torch.Tensor): Shape (batch_size, latent_dim).
        coordinates (torch.Tensor): Coordinates at which function representation
            should be evaluated. Shape (batch_size, *, coordinate_dim).
        features (torch.Tensor): Target features for model to match. Shape
            (batch_size, *, feature_dim).
        inner_steps (int): Number of inner loop steps to take.
        inner_lr (float): Learning rate for inner loop.
        is_train (bool):
        gradient_checkpointing (bool): If True uses gradient checkpointing. This
            can massively reduce memory consumption.
    """
    fitted_modulations = modulations
    for step in range(inner_steps):
        if gradient_checkpointing:
            fitted_modulations = cp.checkpoint(
                inner_loop_step,
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                torch.as_tensor(inner_lr),
                torch.as_tensor(is_train),
                torch.as_tensor(gradient_checkpointing),
            )
        else:
            fitted_modulations = inner_loop_step(
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                inner_lr,
                is_train,
                gradient_checkpointing,
            )
    return fitted_modulations


def inner_loop_step(
    func_rep,
    modulations,
    coordinates,
    features,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
):
    """Performs a single inner loop step."""
    detach = not torch.is_grad_enabled() and gradient_checkpointing
    batch_size = len(features)

    with torch.enable_grad():
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        # Note we multiply by batch size here to undo the averaging across batch
        # elements from the MSE function. Indeed, each set of modulations is fit
        # independently and the size of the gradient should not depend on how
        # many elements are in the batch
        loss = losses.mse_fn(features_recon, features) * batch_size
        # If we are training, we should create graph since we will need this to
        # compute second order gradients in the MAML outer loop
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train and not detach,
        )[0]
    # Perform single gradient descent step
    return modulations - inner_lr * grad


def outer_step(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """
    func_rep.zero_grad()
    batch_size = len(coordinates)
    modulations_init = torch.zeros(
        batch_size, func_rep.modulation_net.latent_dim, device=coordinates.device
    ).requires_grad_()

    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations_init,
        coordinates,
        features,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        # While the loss in the inner loop is individual for each set of
        # modulations, the loss in the outer loop does depend on the entire
        # batch (we update the base network such that all modulations can easily
        # be fit). We therefore take the mean across the batch dimension so the
        # loss is invariant to the number of elements in the batch
        # Shape (batch_size,)
        per_example_loss = losses.batch_mse_fn(features_recon, features)
        # Shape (1,)
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = features_recon

    return outputs


def outer_step_chunked(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    chunk_size,
    gradient_checkpointing=False,
):
    """Performs outer step in chunks to reduce memory requirements when a
    datapoint has a large number of patches.

    Args:
        coordinates (torch.Tensor): Shape (num_patches, *, coordinate_dim).
        features (torch.Tensor): Shape (num_patches, *, feature_dim).
        chunk_size (int): Size of chunks to use when fitting inner loop.
            Typically chunk_size < num_patches in order to reduce memory
            requirements.

    Notes:
        This should only be used for validation, not training. Note also that
        this function should only be used for patching, when a large number
        of patches represents a single datapoint. In other cases, batch size
        can just directly be reduced. This function only returns reconstructions
        and modulations.
    """
    # Calculate number of batches of size chunk_size needed to process datapoint
    num_patches = coordinates.shape[0]
    num_batches = num_patches // chunk_size
    last_batch_size = num_patches % chunk_size

    # Fit patches separately and stack them
    reconstructions = []
    modulations = []
    idx = 0
    for _ in range(num_batches):
        next_idx = idx + chunk_size
        outputs = outer_step(
            func_rep,
            coordinates[idx:next_idx],
            features[idx:next_idx],
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            is_train=False,
            return_reconstructions=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        # Shape (chunk_size, *, feature_dim)
        reconstructions.append(outputs["reconstructions"])
        # Shape (chunk_size, latent_dim)
        modulations.append(outputs["modulations"])
        idx = next_idx

    # If non zero final batch size, fit final piece of data
    if last_batch_size:
        outputs = outer_step(
            func_rep,
            coordinates[idx:],
            features[idx:],
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            is_train=False,
            return_reconstructions=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        reconstructions.append(outputs["reconstructions"])
        modulations.append(outputs["modulations"])

    # Reconstructions shape (num_patches, *, feature_dim)
    # Modulations shape (num_patches, latent_dim)
    return {
        "reconstructions": torch.cat(reconstructions, dim=0),
        "modulations": torch.cat(modulations, dim=0),
    }
