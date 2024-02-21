import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    """ResNet block."""

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        hidden_activation=nn.SiLU(),
        out_activation=nn.SiLU(),
        normalization=nn.Identity(),
    ):
        super().__init__()
        assert in_dim <= out_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.normalization = normalization

        self._network = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            self.hidden_activation,
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, x):
        device = x.device
        output = self._network(x)
        output = output + torch.cat(
            [x, torch.zeros([x.size(dim=0), self.out_dim - self.in_dim]).to(device)],
            dim=1,
        )
        return self.normalization(self.out_activation(output))


class _StandardBlock(nn.Module):
    """Standard fully-connected layer."""

    def __init__(
        self, in_dim, out_dim, activation=nn.ReLU(), normalization=nn.Identity()
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.normalization = normalization

        self._network = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim), self.activation, self.normalization
        )

    def forward(self, x):
        return self._network(x)


class NeRFNetwork(nn.Module):
    """
    Vanila NeRF network. 2 hyperparameters.
        spatial_encoding_l: int, hyperparameter L for spatial positional encoding.
        directional_encoding_l: int, hyperparameter L for directional positional encoding.
    """

    def __init__(
        self,
        spatial_encoding_l,
        directional_encoding_l,
    ):
        super().__init__()
        self.spatial_encoding_l = spatial_encoding_l
        self.directional_encoding_l = directional_encoding_l

        self._density_network1 = nn.Sequential()
        self._density_network2 = nn.Sequential()
        self._color_network = nn.Sequential()

        # Refer to NeRF paper.
        self._density_network1.append(
            _StandardBlock(6 * self.spatial_encoding_l + 3, 256, activation=nn.ReLU())
        )
        for _ in range(4):
            self._density_network1.append(
                _StandardBlock(256, 256, activation=nn.ReLU())
            )

        self._density_network2.append(
            _StandardBlock(6 * self.spatial_encoding_l + 259, 256, activation=nn.ReLU())
        )
        for _ in range(2):
            self._density_network2.append(
                _StandardBlock(256, 256, activation=nn.ReLU())
            )
        self._density_network2.append(
            _StandardBlock(256, 257, activation=nn.Identity())
        )

        self._color_network.append(
            _StandardBlock(
                6 * self.directional_encoding_l + 256, 128, activation=nn.ReLU()
            )
        )
        self._color_network.append(_StandardBlock(128, 3, activation=nn.Sigmoid()))

    def forward(self, x, d):
        """
        Forward process of NeRF network.

        Input
            x: torch.Tensor, size=[N, 6*self.spatial_encoding_l + 3], encoded positions and coords.
            d: torch.Tensor, size=[N, 6*self.directional_encoding_l], encoded directions.
        Output
            density: torch.Tensor, size=[N], volume density.
            color: torch.Tensor, size=[N,3], view-dependent color.
        """
        h = self._density_network1(x)
        h = self._density_network2(torch.cat([x, h], dim=1))
        density = F.relu(h[:, 0])
        color = self._color_network(torch.cat([h[:, 1:], d], dim=1))

        return density, color
