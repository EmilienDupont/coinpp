import torch


mse_fn = torch.nn.MSELoss()
per_element_mse_fn = torch.nn.MSELoss(reduction="none")


def batch_mse_fn(x1, x2):
    """Computes MSE between two batches of signals while preserving the batch
    dimension (per batch element MSE).

    Args:
        x1 (torch.Tensor): Shape (batch_size, *).
        x2 (torch.Tensor): Shape (batch_size, *).

    Returns:
        MSE tensor of shape (batch_size,).
    """
    # Shape (batch_size, *)
    per_element_mse = per_element_mse_fn(x1, x2)
    # Shape (batch_size,)
    return per_element_mse.view(x1.shape[0], -1).mean(dim=1)


def mse2psnr(mse):
    """Computes PSNR from MSE, assuming the MSE was calculated between signals
    lying in [0, 1].

    Args:
        mse (torch.Tensor or float):
    """
    return -10.0 * torch.log10(mse)


def psnr_fn(x1, x2):
    """Computes PSNR between signals x1 and x2. Note that the values of x1 and
    x2 are assumed to lie in [0, 1].

    Args:
        x1 (torch.Tensor): Shape (*).
        x2 (torch.Tensor): Shape (*).
    """
    return mse2psnr(mse_fn(x1, x2))
