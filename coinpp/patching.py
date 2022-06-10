import math
import torch
from typing import Tuple, Union


class Patcher:
    """Class to patch and unpatch data.

    Args:
        patch_shape (tuple of ints). Patch size of audio, image or volume. For example
            (200,) for audio, (64, 64) for an image or (32, 32, 16) for a volume.

    Notes:
        Only works for regular volumetric data, such as images and MRI scans,
        not spherical data such as ERA5.
    """

    def __init__(self, patch_shape):
        assert len(patch_shape) in (1, 2, 3)
        self.patch_shape = patch_shape
        self.patch_dims = len(patch_shape)
        # self.is_3d = len(patch_shape) == 3

    def patch(self, data):
        """Splits data into patches. If the patch shape doesn't divide the data
        shape, use reflection padding.

        Args:
            data (torch.Tensor): Shape (channels, width) or (channels, height, width) or
                (channels, depth, height, width). Note that there should not be
                a batch dimension.

        Returns:
            Patched data of shape (num_patches, channels, {depth, height,} width)
            and a tuple ({depth, height,} width) specifiying the original shape
            of the data (this is required to reconstruct the data).
        """
        if self.patch_dims == 1:
            assert data.ndim == 2, "Incorrect data shape for 1d audio."

            # Extract shapes
            channels = data.shape[0]
            spatial_shape = data.shape[1:]

            # Pad data so it can be divided into equally sized patches
            pad_width = get_padding(spatial_shape, self.patch_shape)
            padding = (0, pad_width)
            padded = torch.nn.functional.pad(data, padding, mode="reflect")

            # padded has shape (channels, padded_width)
            # Reshape to (num_patches, channels, patch_width)
            return padded.reshape(-1, channels, self.patch_shape[0]), spatial_shape
        elif self.patch_dims == 2:
            assert data.ndim == 3, "Incorrect data shape for images."

            # Extract shapes
            channels = data.shape[0]
            spatial_shape = data.shape[1:]
            patch_height, patch_width = self.patch_shape

            # Pad data so it can be divided into equally sized patches
            pad_height, pad_width = get_padding(spatial_shape, self.patch_shape)
            # Note that padding operates from last to first in terms of dimension
            # i.e. (left, right, top, bottom)
            padding = (0, pad_width, 0, pad_height)
            padded = torch.nn.functional.pad(data, padding, mode="reflect")

            # padded has shape (channels, padded_height, padded_width),
            # unsqueeze this to add a batch dimension (expected by unfold)
            patches = torch.nn.functional.unfold(
                padded.unsqueeze(0),
                stride=self.patch_shape,
                kernel_size=self.patch_shape,
            )
            # patches has shape (1, channels * patch_height * patch_width, num_patches).
            # Reshape to (num_patches, channels, patch_height, patch_width)
            patches = patches.reshape(channels, patch_height, patch_width, -1).permute(
                3, 0, 1, 2
            )
            # Return patches and data shape, so data can be reconstructed from
            # patches
            return patches, spatial_shape
        elif self.patch_dims == 3:
            assert data.ndim == 4, "Incorrect data shape for 3d volumes."

            # Extract shapes
            channels = data.shape[0]
            spatial_shape = data.shape[1:]
            patch_depth, patch_height, patch_width = self.patch_shape

            # Pad data so it can be divided into equally sized patches
            pad_depth, pad_height, pad_width = get_padding(
                spatial_shape, self.patch_shape
            )
            # Note that padding operates from last to first in terms of dimension
            # i.e. (left, right, top, bottom, front, back)
            padding = (0, pad_width, 0, pad_height, 0, pad_depth)
            padded = torch.nn.functional.pad(data, padding, mode="reflect")

            # padded has shape (channels, padded_depth, padded_height, padded_width),
            # unsqueeze this to add a batch dimension (expected by unfold)
            patches = unfold3d(
                padded.unsqueeze(0),
                stride=self.patch_shape,
                kernel_size=self.patch_shape,
            )
            # patches has shape (1, channels * pathch_depth * patch_height * patch_width, num_patches).
            # Reshape to (num_patches, channels, patch_height, patch_width)
            patches = patches.reshape(
                channels, patch_depth, patch_height, patch_width, -1
            ).permute(4, 0, 1, 2, 3)
            # Return patches and data shape, so data can be reconstructed from
            # patches
            return patches, spatial_shape

    def unpatch(self, patches, spatial_shape):
        """
        Args:
            patches (torch.Tensor): Shape (num_patches, channels, {patch_depth,
                patch_height,} patch_width).
            spatial_shape (tuple of ints): Tuple describing spatial dims of
                original unpatched data, i.e. ({depth, height,} width).
        """
        if self.patch_dims == 1:
            # Calculate padded shape (required to reshape)
            width = spatial_shape[0]
            pad_width = get_padding(spatial_shape, self.patch_shape)
            padded_width = width + pad_width

            # Reshape patches tensor from shape (num_patches, channels, patch_width),
            # to (channels, padded_width) and remove padding to get tensor of shape
            # (channels, width)
            return patches.reshape(-1, padded_width)[:, :width]
        elif self.patch_dims == 2:
            # Calculate padded shape (required by fold function)
            height, width = spatial_shape
            pad_height, pad_width = get_padding(spatial_shape, self.patch_shape)
            padded_shape = (height + pad_height, width + pad_width)

            # Reshape patches tensor from (num_patches, channels, patch_height, patch_width)
            # to (1, channels * patch_height * patch_width, num_patches)
            num_patches, channels, patch_height, patch_width = patches.shape
            patches = patches.permute(1, 2, 3, 0).reshape(1, -1, num_patches)
            # Fold data to return a tensor of shape (1, channels, padded_height, padded_width)
            padded_data = torch.nn.functional.fold(
                patches,
                output_size=padded_shape,
                kernel_size=self.patch_shape,
                stride=self.patch_shape,
            )

            # Remove padding to get tensor of shape (channels, height, width)
            return padded_data[0, :, :height, :width]
        elif self.patch_dims == 3:
            # Calculate padded shape (required by fold function)
            depth, height, width = spatial_shape
            pad_depth, pad_height, pad_width = get_padding(
                spatial_shape, self.patch_shape
            )
            padded_shape = (depth + pad_depth, height + pad_height, width + pad_width)

            # Reshape patches tensor from (num_patches, channels, patch_depth, patch_height, patch_width)
            # to (1, channels * patch_depth, patch_height * patch_width, num_patches)
            (
                num_patches,
                channels,
                patch_depth,
                patch_height,
                patch_width,
            ) = patches.shape
            patches = patches.permute(1, 2, 3, 4, 0).reshape(1, -1, num_patches)
            # Fold data to return a tensor of shape (1, channels, padded_depth, padded_height, padded_width)
            padded_data = fold3d(
                patches,
                output_size=padded_shape,
                kernel_size=self.patch_shape,
                stride=self.patch_shape,
            )

            # Remove padding to get tensor of shape (channels, depth, height, width)
            return padded_data[0, :, :depth, :height, :width]


def get_padding(spatial_shape, patch_shape):
    """Returns padding required to make patch_shape divide data_shape into equal
    patches.

    Args:
        spatial_shape (tuple of ints): Shape ({depth, height,} width).
        patch_shape (tuple of ints): Shape ({patch_depth, patch_height,} patch_width).
    """
    if len(patch_shape) == 1:
        patch_width = patch_shape[0]
        width = spatial_shape[0]
        excess_width = width % patch_width
        pad_width = patch_width - excess_width if excess_width else 0
        return pad_width
    if len(patch_shape) == 2:
        patch_height, patch_width = patch_shape
        height, width = spatial_shape
        excess_height = height % patch_height
        excess_width = width % patch_width
        pad_height = patch_height - excess_height if excess_height else 0
        pad_width = patch_width - excess_width if excess_width else 0
        return pad_height, pad_width
    elif len(patch_shape) == 3:
        patch_depth, patch_height, patch_width = patch_shape
        depth, height, width = spatial_shape
        excess_depth = depth % patch_depth
        excess_height = height % patch_height
        excess_width = width % patch_width
        pad_depth = patch_depth - excess_depth if excess_depth else 0
        pad_height = patch_height - excess_height if excess_height else 0
        pad_width = patch_width - excess_width if excess_width else 0
        return pad_depth, pad_height, pad_width


def unfold3d(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    padding: Union[int, Tuple[int, int, int]] = 0,
    stride: Union[int, Tuple[int, int, int]] = 1,
    dilation: Union[int, Tuple[int, int, int]] = 1,
):
    """
    Extracts sliding local blocks from an batched input tensor.
    :class:`torch.nn.Unfold` only supports 4D inputs (batched image-like tensors).
    This method implements the same action for 5D inputs

    Args:
        tensor: An input tensor of shape ``(B, C, D, H, W)``.
        kernel_size: the size of the sliding blocks
        padding: implicit zero padding to be added on both sides of input
        stride: the stride of the sliding blocks in the input spatial dimensions

    Example:
        >>> B, C, D, H, W = 3, 4, 5, 6, 7
        >>> tensor = torch.arange(1,B*C*D*H*W+1.).view(B,C,D,H,W)
        >>> unfold3d(tensor, kernel_size=2, padding=0, stride=1).shape
        torch.Size([3, 32, 120])

    Returns:
        A tensor of shape ``(B, C * np.product(kernel_size), L)``, where L - output spatial dimensions.
        See :class:`torch.nn.Unfold` for more details

    Notes:
        This function was copied (and slightly modified) from the opacus library
        https://opacus.ai/api/_modules/opacus/utils/tensor_utils.html#unfold3d
        which is licensed under the Apache License 2.0.
    """
    if len(tensor.shape) != 5:
        raise ValueError(
            f"Input tensor must be of the shape [B, C, D, H, W]. Got{tensor.shape}"
        )

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)

    if isinstance(padding, int):
        padding = (padding, padding, padding)

    if isinstance(stride, int):
        stride = (stride, stride, stride)

    batch_size, channels, _, _, _ = tensor.shape

    # Shape (B, C, D, H, W) -> (B, C, D+2*padding[2], H+2*padding[1], W+2*padding[0])
    tensor = torch.nn.functional.pad(
        tensor, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
    )

    # Output shape: (B, C, D_out, H_out, W_out, kernel_size[0], kernel_size[1], kernel_size[2])
    # For D_out, H_out, W_out definitions see :class:`torch.nn.Unfold`
    tensor = tensor.unfold(dimension=2, size=kernel_size[0], step=stride[0])
    tensor = tensor.unfold(dimension=3, size=kernel_size[1], step=stride[1])
    tensor = tensor.unfold(dimension=4, size=kernel_size[2], step=stride[2])

    # Output shape: (B, D_out, H_out, W_out, C, kernel_size[0], kernel_size[1], kernel_size[2])
    tensor = tensor.permute(0, 2, 3, 4, 1, 5, 6, 7)

    # Output shape: (B, D_out * H_out * W_out, C * kernel_size[0] * kernel_size[1] * kernel_size[2]
    tensor = tensor.reshape(
        batch_size, -1, channels * math.prod(kernel_size)
    ).transpose(1, 2)

    return tensor


def fold3d(patches, output_size, kernel_size, stride):
    """Equivalent of torch.nn.functional.fold for 3D data (i.e. 5D tensors when
    counting batch and channel dimensions).

    Args:
        patches (torch.Tensor): Tensor of shape (1,
            channels * kernel_size[0] * kernel_size[1] * kernel_size[2], num_patches).
        output_size (tuple of int): The shape of the spatial dimensions of the
            output, i.e. (depth, height, width).
        kernel_size (tuple of int): The size of the sliding blocks
        stride (tuple of int): The stride of the sliding blocks in the input
            spatial dimensions

    Returns:
        Tensor of shape (1, channels, depth, height, width).

    Notes:
        The batch dimension must be 1.
    """
    depth, height, width = output_size
    # As patches.shape[1] = channels * kernel_size[0] * kernel_size[1] * kernel_size[2],
    # extract number of channels
    channels = patches.shape[1] // math.prod(kernel_size)
    output = torch.zeros(1, channels, *output_size, device=patches.device)

    # Iterate over patches and fill out the unpatched version
    depth_prev, depth_next = 0, 0
    height_prev, height_next = 0, 0
    width_prev, width_next = 0, 0

    for i in range(patches.shape[2]):
        # Extract patch of shape (channels * patch_depth * patch_height * patch_width,)
        # and reshape
        patch = patches[:, :, i]
        patch = patch.reshape(channels, kernel_size[0], kernel_size[1], kernel_size[2])

        if not (width_prev == width - kernel_size[2]):
            width_next = width_next + stride[2]
        elif height_prev == height - kernel_size[1]:
            width_next = 0
            height_next = 0
            depth_next = depth_prev + stride[0]
        else:
            width_next = 0
            height_next = height_prev + stride[1]

        # Add patch to output
        output[
            :,
            :,
            depth_prev : depth_prev + kernel_size[0],
            height_prev : height_prev + kernel_size[1],
            width_prev : width_prev + kernel_size[2],
        ] += patch

        depth_prev = depth_next
        height_prev = height_next
        width_prev = width_next

    # Shape (1, channels, depth, height, width)
    return output


if __name__ == "__main__":
    # 1D tests
    patch_shape = (20,)
    data = torch.rand((2, 105))
    patcher = Patcher(patch_shape)
    patched, data_shape = patcher.patch(data)
    print(data.shape)
    print(patched.shape)

    data_unpatched = patcher.unpatch(patched, data_shape)
    print(data_unpatched.shape)

    print((data - data_unpatched).abs().sum())

    # 2D tests
    patch_shape = (20, 10)
    data = torch.rand((3, 95, 100))
    patcher = Patcher(patch_shape)
    patched, data_shape = patcher.patch(data)
    print(data.shape)
    print(patched.shape)

    data_unpatched = patcher.unpatch(patched, data_shape)
    print(data_unpatched.shape)

    print((data - data_unpatched).abs().sum())

    # 3D tests
    patch_shape = (4, 5, 8)
    data = torch.rand((1, 8, 8, 9))
    patcher = Patcher(patch_shape)
    patched, data_shape = patcher.patch(data)
    print(data.shape)
    print(patched.shape)

    data_unpatched = patcher.unpatch(patched, data_shape)
    print(data_unpatched.shape)

    print((data - data_unpatched).abs().sum())
