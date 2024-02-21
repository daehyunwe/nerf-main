from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np

import src.graphics.camera as cam
import src.graphics.world as wld
import src.renderer.encoder as enc
import src.renderer.sampler as sam


def _volume_renderer(
    density: torch.Tensor,
    color: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Perform volume rendering rays.

    Input
        density: size=[N, NS]
        color: size=[N, NS, 3]
        t: size=[N, NS] or [N, NS + 1]
    Output
        pixel_color: size=[N, 3]
    """

    N = density.size(dim=0)
    NS = density.size(dim=1)

    if t.size(dim=1) == NS:
        # compute delta, alpha, transmittance, and weight
        delta = F.relu(t[:, 1:] - t[:, :-1])  # [N, NS - 1]
        alpha = torch.ones_like(density[:, :-1]) - torch.exp(
            -density[:, :-1] * delta
        )  # [N, NS - 1]
        accumulated_transmittance = torch.exp(
            -torch.cat(
                [
                    torch.zeros(N, 1),
                    torch.cumsum(density[:, :-1] * delta, dim=1),
                ],
                dim=1,
            )[:, :-1]
        )  # [N, NS - 1]
        weight = accumulated_transmittance * alpha  # [N, NS - 1]

        # integrate
        pixel_color = (
            weight.view(N, NS - 1, 1).expand(-1, -1, 3) * color[:, :-1, :]
        ).sum(
            dim=1
        )  # [N, 3]

    elif t.size(dim=1) == NS + 1:
        # compute delta, alpha, transmittance, and weight
        delta = F.relu(t[:, 1:] - t[:, :-1])  # [N, NS]
        alpha = torch.ones_like(density) - torch.exp(-density * delta)  # [N, NS]
        accumulated_transmittance = torch.exp(
            -torch.cat(
                [
                    torch.zeros(N, 1),
                    torch.cumsum(density * delta, dim=1),
                ],
                dim=1,
            )[:, :-1]
        )  # [N, NS]
        weight = accumulated_transmittance * alpha  # [N, NS]

        # integrate
        pixel_color = (weight.view(N, NS, 1).expand(-1, -1, 3) * color).sum(
            dim=1
        )  # [N, 3]

    else:
        raise AssertionError

    return pixel_color


def render_pixels(
    model: dict,  # input
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    pixel_density: torch.Tensor,
    num_coarse_samples: int,  # hyperparameters
    num_fine_samples: int,
    spatial_encoding_l: int,
    directional_encoding_l: int,
    near_dist: float,
    far_dist: float,
    bounding_volume_size: float,
    weight_filtering_alpha: float,
    nerf_type: str = "vanila",  # conifgs
    render_type: str = "albedo",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render pixels.

    Input
        model: dictionary of nerf networks.
        ray_origins: size=[N, 3]
        ray_directions: size=[N, 3]
        pixel_density: size=[2], pixel density [m_x, m_y] of camera.

        num_coarse_samples: number of coarse samples.
        num_fine_samples: number of fine samples.
        spatial_encoding_l: hyperparameter for controlling maximum spatial encoding frequency.
        directional_encoding_l: hyperparameter for controlling maximum directional encoding frequency.
        near_dist: near plane distance.
        far_dist: far plane distance.
        bounding_volume_size: size of the bounding volume.
        weight_filtering_alpha: hyperparameter from Mip-NeRF.

        nerf_type: str, "vanila" or "mip" or "dreamfusion".
        render_type: str, "shaded" or "albedo" or "textureless".

    Output
        coarse_pixel_color: size=[N,3].
        union_pixel_color: size=[N,3].
    """

    device = ray_origins.device

    if nerf_type == "vanila":
        if render_type == "shaded":
            raise NotImplementedError

        elif render_type == "albedo":
            # dimensions
            N = ray_origins.size(dim=0)
            NCS = num_coarse_samples
            NFS = num_fine_samples

            # interpret inputs
            coarse_network = model["coarse_network"]
            fine_network = model["fine_network"]

            # scan the edges of cube
            scan_t = sam.coarse_t_sampler(N, NCS, near_dist, far_dist)  # [N, NCS]
            scan_points = ray_origins.view(N, 1, 3).expand(-1, NCS, -1) + scan_t.view(
                N, NCS, 1
            ).expand(-1, -1, 3) * ray_directions.view(N, 1, 3).expand(
                -1, NCS, -1
            )  # [N, NCS, 3]
            isin = torch.logical_and(
                scan_points >= -bounding_volume_size,
                scan_points <= bounding_volume_size,
            ).all(
                dim=2
            )  # [N, NCS]
            in_scan_t = scan_t * isin  # [N, NCS]
            far_edge_dist, _ = torch.max(in_scan_t, dim=1)  # [N]
            in_scan_t_no_zero = in_scan_t + far_edge_dist.view(N, 1).expand(
                -1, NCS
            ) * torch.logical_not(
                isin
            )  # [N, NCS]
            near_edge_dist, _ = torch.min(in_scan_t_no_zero, dim=1)  # [N]
            far_edge_dist = far_edge_dist[far_edge_dist.nonzero(as_tuple=True)]
            near_edge_dist = near_edge_dist[near_edge_dist.nonzero(as_tuple=True)]

            # select rays to be rendered
            rays_to_render = isin.any(dim=1)  # [N]
            rays_to_render_idx = rays_to_render.nonzero()  # [N, 1]
            ray_origins_to_render = ray_origins[rays_to_render]  # [M, 3]
            ray_directions_to_render = ray_directions[rays_to_render]  # [M, 3]
            M = rays_to_render.sum(dim=0)

            # sample coarse points
            coarse_t = sam.coarse_t_sampler(
                M, NCS, near_edge_dist, far_edge_dist
            )  # [M, NCS]
            coarse_points = ray_origins_to_render.view(M, 1, 3).expand(
                -1, NCS, -1
            ) + coarse_t.view(M, NCS, 1).expand(
                -1, -1, 3
            ) * ray_directions_to_render.view(
                M, 1, 3
            ).expand(
                -1, NCS, -1
            )  # [M, NCS, 3]

            # encoding
            encoded_coarse_points = enc.positional_encode(
                coarse_points.view(M * NCS, 3), spatial_encoding_l
            ).view(
                M, NCS, 6 * spatial_encoding_l
            )  # [M, NCS, 6*L]
            encoded_directions = enc.positional_encode(
                ray_directions_to_render, directional_encoding_l
            )  # [M, 6*L]

            # coarse forward pass
            coarse_density, coarse_color = coarse_network(
                torch.cat(
                    [
                        encoded_coarse_points.view(M * NCS, 6 * spatial_encoding_l),
                        coarse_points.view(M * NCS, 3),
                    ],
                    dim=1,
                ),
                torch.repeat_interleave(
                    encoded_directions,
                    NCS,
                    dim=0,
                ),
            )
            coarse_density = coarse_density.view(M, NCS)  # [M, NCS]
            coarse_color = coarse_color.view(M, NCS, 3)  # [M, NCS, 3]

            # volume rendering on coarse points
            coarse_pixel_color_rendered = _volume_renderer(
                coarse_density,
                coarse_color,
                coarse_t,
            )  # [M, 3]
            coarse_pixel_color = torch.zeros([N, 3], device=device).scatter(
                0, rays_to_render_idx.expand(-1, 3), coarse_pixel_color_rendered
            )  # [N, 3]

            # fine sampling
            fine_t = sam.fine_t_sampler(
                coarse_density,
                coarse_t,
                NFS,
                weight_filtering_alpha,
            )  # [M, NFS]

            # obtain union of coarse and fine samples
            union_t, _ = torch.cat([coarse_t, fine_t], dim=1).sort(
                dim=1
            )  # [M, NCS + NFS]
            union_points = ray_origins_to_render.view(M, 1, 3).expand(
                -1, NCS + NFS, -1
            ) + union_t.view(M, NCS + NFS, 1).expand(
                -1, -1, 3
            ) * ray_directions_to_render.view(
                M, 1, 3
            ).expand(
                -1, NCS + NFS, -1
            )  # [M, NCS + NFS, 3]

            # encoding
            encoded_union_points = enc.positional_encode(
                union_points.view(M * (NCS + NFS), 3), spatial_encoding_l
            ).view(
                M, NCS + NFS, 6 * spatial_encoding_l
            )  # [M, NCS + NFS, 6*L]

            # fine forward pass
            union_density, union_color = fine_network(
                torch.cat(
                    [
                        encoded_union_points.view(
                            M * (NCS + NFS), 6 * spatial_encoding_l
                        ),
                        union_points.view(M * (NCS + NFS), 3),
                    ],
                    dim=1,
                ),
                torch.repeat_interleave(encoded_directions, NCS + NFS, dim=0),
            )
            union_density = union_density.view(M, NCS + NFS)  # [M, NCS + NFS]
            union_color = union_color.view(M, NCS + NFS, 3)  # [M, NCS + NFS, 3]

            # volume rendering on union samples
            union_pixel_color_rendered = _volume_renderer(
                union_density,
                union_color,
                union_t,
            )  # [M, 3]

            union_pixel_color = torch.zeros([N, 3], device=device).scatter(
                0, rays_to_render_idx.expand(-1, 3), union_pixel_color_rendered
            )  # [N, 3]

            return coarse_pixel_color, union_pixel_color

        elif render_type == "textureless":
            raise NotImplementedError

        else:
            raise NotImplementedError

    elif nerf_type == "mip":
        if render_type == "shaded":
            raise NotImplementedError

        elif render_type == "albedo":
            # dimensions
            N = ray_origins.size(dim=0)
            NCS = num_coarse_samples
            NFS = num_fine_samples

            # interpret inputs
            network = model["network"]

            # scan the edges of cube
            scan_t = sam.coarse_t_sampler(N, NCS, near_dist, far_dist)  # [N, NCS]
            scan_points = ray_origins.view(N, 1, 3).expand(-1, NCS, -1) + scan_t.view(
                N, NCS, 1
            ).expand(-1, -1, 3) * ray_directions.view(N, 1, 3).expand(
                -1, NCS, -1
            )  # [N, NCS, 3]
            isin = torch.logical_and(
                scan_points >= -bounding_volume_size,
                scan_points <= bounding_volume_size,
            ).all(
                dim=2
            )  # [N, NCS]
            in_scan_t = scan_t * isin  # [N, NCS]
            far_edge_dist, _ = torch.max(in_scan_t, dim=1)  # [N]
            in_scan_t_no_zero = in_scan_t + far_edge_dist.view(N, 1).expand(
                -1, NCS
            ) * torch.logical_not(
                isin
            )  # [N, NCS]
            near_edge_dist, _ = torch.min(in_scan_t_no_zero, dim=1)  # [N]
            far_edge_dist = far_edge_dist[far_edge_dist.nonzero(as_tuple=True)]
            near_edge_dist = near_edge_dist[near_edge_dist.nonzero(as_tuple=True)]

            # select rays to be rendered
            rays_to_render = isin.any(dim=1)  # [N]
            rays_to_render_idx = rays_to_render.nonzero()  # [N, 1]
            ray_origins_to_render = ray_origins[rays_to_render]  # [M, 3]
            ray_directions_to_render = ray_directions[rays_to_render]  # [M, 3]
            M = rays_to_render.sum(dim=0)

            # sample coarse points
            coarse_t = sam.coarse_t_sampler(
                M, NCS + 1, near_edge_dist, far_edge_dist
            )  # [M, NCS + 1]

            # encoding
            cone_radius = 2 / np.sqrt(12) / pixel_density[0]
            encoded_coarse_points, coarse_means = enc.integrated_positional_encode(
                ray_origins_to_render,
                ray_directions_to_render,
                coarse_t,
                cone_radius,
                spatial_encoding_l,
            )  # [M, NCS, 6*L]
            encoded_directions = enc.positional_encode(
                ray_directions_to_render, directional_encoding_l
            )  # [M, 6*L]

            # coarse forward pass
            coarse_density, coarse_color = network(
                torch.cat(
                    [
                        encoded_coarse_points.view(M * NCS, 6 * spatial_encoding_l),
                        coarse_means.view(M * NCS, 3),
                    ],
                    dim=1,
                ),
                torch.repeat_interleave(
                    encoded_directions,
                    NCS,
                    dim=0,
                ),
            )
            coarse_density = coarse_density.view(M, NCS)  # [M, NCS]
            coarse_color = coarse_color.view(M, NCS, 3)  # [M, NCS, 3]

            # volume rendering on coarse points
            coarse_pixel_color_rendered = _volume_renderer(
                coarse_density,
                coarse_color,
                coarse_t,
            )  # [M, 3]
            coarse_pixel_color = torch.zeros([N, 3], device=device).scatter(
                0, rays_to_render_idx.expand(-1, 3), coarse_pixel_color_rendered
            )  # [N, 3]

            # fine sampling
            fine_t = sam.fine_t_sampler(
                coarse_density,
                coarse_t,
                NFS + 1,
                weight_filtering_alpha,
            )  # [M, NFS + 1]

            # encoding
            encoded_fine_points, fine_means = enc.integrated_positional_encode(
                ray_origins_to_render,
                ray_directions_to_render,
                fine_t,
                cone_radius,
                spatial_encoding_l,
            )  # [M, NFS, 6*L]

            # fine forward pass
            fine_density, fine_color = network(
                torch.cat(
                    [
                        encoded_fine_points.view(M * NFS, 6 * spatial_encoding_l),
                        fine_means.view(M * NFS, 3),
                    ],
                    dim=1,
                ),
                torch.repeat_interleave(encoded_directions, NFS, dim=0),
            )
            fine_density = fine_density.view(M, NFS)  # [M, NFS]
            fine_color = fine_color.view(M, NFS, 3)  # [M, NFS, 3]

            # volume rendering on fine samples
            fine_pixel_color_rendered = _volume_renderer(
                fine_density,
                fine_color,
                fine_t,
            )  # [M, 3]

            fine_pixel_color = torch.zeros([N, 3], device=device).scatter(
                0, rays_to_render_idx.expand(-1, 3), fine_pixel_color_rendered
            )  # [N, 3]

            return coarse_pixel_color, fine_pixel_color

        elif render_type == "textureless":
            raise NotImplementedError

        else:
            raise NotImplementedError

    elif nerf_type == "dreamfusion":
        if render_type == "shaded":
            raise NotImplementedError

        elif render_type == "albedo":
            raise NotImplementedError

        elif render_type == "textureless":
            raise NotImplementedError

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError


def render_image(
    model: dict,  # input
    world: wld.World,
    camera_idx: int,
    num_coarse_samples: int,  # hyperparameters
    num_fine_samples: int,
    spatial_encoding_l: int,
    directional_encoding_l: int,
    weight_filtering_alpha: float,
    bounding_volume_size: float,
    nerf_type: str = "vanila",  # conifgs
    render_type: str = "albedo",
) -> torch.Tensor:
    """
    Render the full image seen from the given camera.

    Input
        model: dictionary of nerf networks.
        world: cameras and light sources.
        camera_idx: index of the camera in cameras list.

        num_coarse_samples: number of coarse samples.
        num_fine_samples: number of fine samples.
        spatial_encoding_l: hyperparameter for controlling maximum spatial encoding frequency.
        directional_encoding_l: hyperparameter for controlling maximum directional encoding frequency.
        bounding_volume_size: size of the bounding volume.
        weight_filtering_alpha: hyperparameter from Mip-NeRF.

        nerf_type: str, "vanila" or "mip" or "dreamfusion".
        render_type: str, "shaded" or "albedo" or "textureless".

    Output
        img: size=[H,W,3]
    """

    camera: cam.Camera = world.cameras[camera_idx]
    H, W = camera.img_size
    x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    pixel_density = camera.pixel_density
    pixel_coordinates = torch.stack([x, y], dim=2).view(H * W, 2)
    ray_origins = camera.position  # [3]
    ray_directions = camera.pixels_to_ray_directions(pixel_coordinates).view(
        H, W, 3
    )  # [H, W, 3]
    color_list = []
    for i in range(H):
        _, color = render_pixels(
            model,
            ray_origins.view(1, 3).expand(W, -1),
            ray_directions[i],
            pixel_density,
            num_coarse_samples,
            num_fine_samples,
            spatial_encoding_l,
            directional_encoding_l,
            camera.near_dist,
            camera.far_dist,
            bounding_volume_size,
            weight_filtering_alpha,
            nerf_type=nerf_type,
            render_type=render_type,
        )
        color_list.append(color)
    img = torch.cat(color_list, dim=0)
    return img.view(H, W, 3)
