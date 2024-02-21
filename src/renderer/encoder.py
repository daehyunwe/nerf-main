import numpy as np
import torch


def integrated_positional_encode(
    origins: torch.Tensor,
    directions: torch.Tensor,
    t: torch.Tensor,
    cone_radius: float,
    L: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Refer section 3.1 in Mip-NeRF paper. Only used for spatial encoding.

    Input
        origins: size=[M, 3]
        directions: size=[M, 3]
        t: size=[M, NS+1]
        cone_radius: Cone radius at image plane.
        L: Hyperparameter representing maximum frequency.
    Output
        ipe_features: size=[M, NS, 6*L]
        means: size=[M, NS, 3]
    """
    M = origins.size(dim=0)
    NS = t.size(dim=1) - 1

    # t_0: [M, NS]
    t_0 = t[:, :-1]
    # t_1: [M, NS]
    t_1 = t[:, 1:]
    t_mu = (t_0 + t_1) / 2
    t_delta = (t_1 - t_0) / 2
    # mean_t: [M, NS]
    mean_t = t_mu + (2 * t_mu * t_delta**2) / (3 * t_mu**2 + t_delta**2)
    # variance_t: [M, NS]
    variance_t = 1 / 3 * t_delta**2 - (
        (4 * t_delta**2**2 * (12 * t_mu**2 - t_delta**2))
        / (15 * (3 * t_mu**2 + t_delta**2) ** 2)
    )
    # variance_r: [M, NS]
    variance_r = cone_radius**2 * (
        1 / 4 * t_mu**2
        + 5 / 12 * t_delta**2
        - (4 * t_delta**2**2) / (15 * (3 * t_mu**2 + t_delta**2))
    )
    # means: [M, NS, 3]
    means = origins.view(M, 1, 3).expand(-1, NS, -1) + mean_t.view(M, NS, 1).expand(
        -1, -1, 3
    ) * directions.view(M, 1, 3).expand(-1, NS, -1)
    # diag_variances: [M, NS, 3]
    diag_variances = variance_t.view(M, NS, 1).expand(-1, -1, 3) * (
        directions**2
    ).view(M, 1, 3).expand(-1, NS, -1) + variance_r.view(M, NS, 1).expand(-1, -1, 3) * (
        torch.ones_like(directions)
        - directions**2
        / torch.sum(directions**2, dim=1, keepdim=True).expand(-1, 3)
    ).view(
        M, 1, 3
    ).expand(
        -1, NS, -1
    )
    # P: [3*L, 3]
    P = torch.arange(0, L)
    P = torch.repeat_interleave(2**P, 3) * torch.tensor([1, 0, 0]).repeat(L)
    P = torch.cat([torch.zeros(2), P])
    P = torch.cat([P[2:].view(-1, 1), P[1:-1].view(-1, 1), P[:-2].view(-1, 1)], dim=1)
    # lifted_means: [M, NS, 3*L]
    lifted_means = torch.einsum("ij,bcj->bci", P, means)
    # diag_lifted_variances: [M, NS, 3*L]
    diag_lifted_variances = diag_variances.repeat(1, 1, L) * torch.repeat_interleave(
        (4 ** torch.arange(L)).view(1, 1, L).expand(M, NS, -1), 3, dim=2
    )
    # ipe_features: [M, NS, 6*L]
    ipe_features = torch.cat(
        [
            torch.sin(lifted_means) * torch.exp(-0.5 * diag_lifted_variances),
            torch.cos(lifted_means) * torch.exp(-0.5 * diag_lifted_variances),
        ],
        dim=2,
    )

    return ipe_features, means


def positional_encode(input: torch.Tensor, L: int) -> torch.Tensor:
    """
    Refer Section 5.1 in NeRF paper.

    Input
        input: size=[N, 3]
        L: the hyperparameter for controlling maximum encoding frequency.
    Output
        pe_features: size=[N, 6*L]
    """
    N = input.size(dim=0)

    frequencies = input.repeat(1, L) * torch.repeat_interleave(
        (2 ** torch.arange(L) * np.pi).view(1, L).expand(N, -1), 3, dim=1
    )
    pe_features = torch.cat([torch.sin(frequencies), torch.cos(frequencies)], dim=1)

    return pe_features
