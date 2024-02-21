import torch

""" General geometric rules for points, rays, and regions """


def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    """
    Input
        points: size=[N, D]
    Output
        (result): size=[N, D+1]
    """
    device = points.device
    N = points.size(dim=0)
    return torch.cat([points, torch.full([N, 1], 1, device=device)], dim=1)


def to_non_homogeneous(points: torch.Tensor) -> torch.Tensor:
    """
    Input
        points: size=[N, D]
    Output
        (result): size=[N, D-1]
    """
    N, D = points.size()
    return points[:, :-1] / points[:, -1].view(N, 1).expand(-1, D - 1)
