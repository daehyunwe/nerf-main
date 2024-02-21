import torch
import torch.nn.functional as F


def coarse_t_sampler(
    num_rays: int | None,
    num_coarse_samples: int,
    near_dist: float | torch.Tensor,
    far_dist: float | torch.Tensor,
) -> torch.Tensor:
    """
    Stratified sampling method.

    Input
        num_rays: N rays to sample from.
        num_coarse_samples: NCS t's are sampled for each ray.
        near_dist: size=[N], near plane dist for each ray.
        far_dist: size=[N], far plane dist for each ray.
    Output
        coarse_t: [N, NCS]
    """
    if (
        isinstance(near_dist, float)
        and isinstance(far_dist, float)
        and isinstance(num_rays, int)
    ):
        N = num_rays
        NCS = num_coarse_samples

        base = (
            torch.linspace(near_dist, far_dist, NCS + 1)[:-1].view(1, NCS).expand(N, -1)
        )
        coarse_t = base + (far_dist - near_dist) / NCS * torch.rand_like(base)

    elif isinstance(near_dist, torch.Tensor) and isinstance(far_dist, torch.Tensor):
        N = near_dist.size(dim=0)
        NCS = num_coarse_samples

        base = torch.linspace(0, 1, NCS + 1)[:-1].view(1, NCS).expand(N, -1)
        base = (
            far_dist.view(N, 1).expand(-1, NCS) - near_dist.view(N, 1).expand(-1, NCS)
        ) * base + near_dist.view(N, 1).expand(-1, NCS)
        coarse_t = base + (
            far_dist.view(N, 1).expand(-1, NCS) - near_dist.view(N, 1).expand(-1, NCS)
        ) / NCS * torch.rand_like(base)

    else:
        raise AssertionError

    return coarse_t


def fine_t_sampler(
    density: torch.Tensor,
    t: torch.Tensor,
    num_fine_samples: int,
    weight_filtering_alpha: float,
) -> torch.Tensor:
    """
    Samples fine_t of multiple camera ray.
    Does not use inverse transform sampling.

    Input
        density: size=[N, NCS]
        t: size=[N, NCS] or [N, NCS + 1]
        num_fine_samples: number of fine samples drawn.
        weight_filtering_alpha: hyperparameter from Mip-NeRF.
    Output
        full_fine_t: size=[N, NFS]
    """

    device = density.device
    N = density.size(dim=0)
    NCS = density.size(dim=1)
    NFS = num_fine_samples

    if t.size(dim=1) == NCS:
        # compute delta, alpha, transmittance, and weight
        delta = F.relu(t[:, 1:] - t[:, :-1])  # [N, NCS - 1]
        alpha = torch.ones_like(density[:, :-1]) - torch.exp(
            -density[:, :-1] * delta
        )  # [N, NCS - 1]
        accumulated_transmittance = torch.exp(
            -torch.cat(
                [
                    torch.zeros(N, 1),
                    torch.cumsum(density[:, :-1] * delta, dim=1),
                ],
                dim=1,
            )[:, :-1]
        )  # [N, NCS - 1]
        weight = accumulated_transmittance * alpha  # [N, NCS - 1]

        # avoid zero weight
        for i in range(N):
            if torch.all(weight[i] == 0):
                weight[i] = weight[i] + torch.ones_like(weight[i])

        # weight filtering from Mip-NeRF
        if not weight_filtering_alpha is None:
            weight_before = torch.cat(
                [weight[:, 0].view(N, 1), weight[:, :-1]], dim=1
            )  # [N, NCS - 1]
            weight_after = torch.cat(
                [weight[:, 1:], weight[:, -1].view(N, 1)], dim=1
            )  # [N, NCS - 1]
            weight = 0.5 * (
                torch.maximum(weight_before, weight)
                + torch.maximum(weight, weight_after)
            ) + torch.full([N, NCS - 1], weight_filtering_alpha)

        # sample base
        base, _ = (
            torch.multinomial(weight, NFS, replacement=True).to(torch.float).sort()
        )  # [N, NFS]

        # scaling TODO: implement inverse transform sampling
        near_t = t[:, 0]  # [N]
        far_t = t[:, -1]  # [N]
        fine_t = near_t.view(N, 1).expand(-1, NFS) + (far_t - near_t).view(N, 1).expand(
            -1, NFS
        ) / (NCS - 1) * (base + torch.rand_like(base))

        # sorting
        full_fine_t, _ = fine_t.sort(dim=1)  # [N, NFS]

    elif t.size(dim=1) == NCS + 1:
        # compute delta, alpha, transmittance, and weight
        delta = F.relu(t[:, 1:] - t[:, :-1])  # [N, NCS]
        alpha = torch.ones_like(density) - torch.exp(-density * delta)  # [N, NCS]
        accumulated_transmittance = torch.exp(
            -torch.cat(
                [
                    torch.zeros(N, 1),
                    torch.cumsum(density * delta, dim=1),
                ],
                dim=1,
            )[:, :-1]
        )  # [N, NCS]
        weight = accumulated_transmittance * alpha  # [N, NCS]

        # avoid zero weight
        for i in range(N):
            if torch.all(weight[i] == 0):
                weight[i] = weight[i] + torch.ones_like(weight[i])

        # weight filtering from Mip-NeRF
        if not weight_filtering_alpha is None:
            weight_before = torch.cat(
                [weight[:, 0].view(N, 1), weight[:, :-1]], dim=1
            )  # [N, NCS]
            weight_after = torch.cat(
                [weight[:, 1:], weight[:, -1].view(N, 1)], dim=1
            )  # [N, NCS]
            weight = 0.5 * (
                torch.maximum(weight_before, weight)
                + torch.maximum(weight, weight_after)
            ) + torch.full([N, NCS], weight_filtering_alpha)

        # sample base
        base, _ = (
            torch.multinomial(weight, NFS, replacement=True).to(torch.float).sort()
        )  # [N, NFS]

        # scaling TODO: implement inverse transform sampling
        near_t = t[:, 0]  # [N]
        far_t = t[:, -1]  # [N]
        fine_t = near_t.view(N, 1).expand(-1, NFS) + (far_t - near_t).view(N, 1).expand(
            -1, NFS
        ) / NCS * (base + torch.rand_like(base))

        # sorting
        full_fine_t, _ = fine_t.sort(dim=1)  # [N, NFS]

    else:
        raise NotImplementedError

    return full_fine_t
