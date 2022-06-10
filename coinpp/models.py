# Based on https://github.com/EmilienDupont/coin
import torch
from torch import nn
from math import sqrt


class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        is_last (bool): Whether this is last layer of model. If it is, no
            activation is applied and 0.5 is added to the output. Since we
            assume all training data lies in [0, 1], this allows for centering
            the output of the model.
        use_bias (bool): Whether to learn bias in linear layer.
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        is_last=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        if self.is_last:
            # We assume target data is in [0, 1], so adding 0.5 allows us to learn
            # zero-centered features
            out += 0.5
        else:
            out = self.activation(out)
        return out


class Siren(nn.Module):
    """SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers

        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(
                SirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )

        self.net = nn.Sequential(*layers)

        self.last_layer = SirenLayer(
            dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias, is_last=True
        )

    def forward(self, x):
        """Forward pass of SIREN model.

        Args:
            x (torch.Tensor): Tensor of shape (*, dim_in), where * means any
                number of dimensions.

        Returns:
            Tensor of shape (*, dim_out).
        """
        x = self.net(x)
        return self.last_layer(x)


class ModulatedSiren(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial

        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx : idx + self.dim_hidden].unsqueeze(1) + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[
                    :, mid_idx + idx : mid_idx + idx + self.dim_hidden
                ].unsqueeze(1)
            else:
                shift = 0.0

            x = module.linear(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_layer(x)
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class LatentToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.

    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(self, latent_dim, num_modulations, dim_hidden, num_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, dim_hidden), nn.ReLU()]
            if num_layers > 2:
                for i in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden), nn.ReLU()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)


class Bias(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size), requires_grad=True)
        # Add latent_dim attribute for compatibility with LatentToModulation model
        self.latent_dim = size

    def forward(self, x):
        return x + self.bias


if __name__ == "__main__":
    dim_in, dim_hidden, dim_out, num_layers = 2, 5, 3, 4
    batch_size, latent_dim = 3, 7
    model = ModulatedSiren(
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        modulate_scale=True,
        use_latent=True,
        latent_dim=latent_dim,
    )
    print(model)
    latent = torch.rand(batch_size, latent_dim)
    x = torch.rand(batch_size, 5, 5, 2)
    out = model(x)
    out = model.modulated_forward(x, latent)
    print(out.shape)
