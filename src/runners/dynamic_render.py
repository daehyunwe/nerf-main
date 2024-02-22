from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import imageio.v3 as iio

import src.graphics.camera as cam
import src.network.network as net
import src.renderer.renderer as rend
import utils.blender_dataset as bldr
import utils.logger as log


def dynamic_render(
    log_path: Path,  # path
    data_path: Path,
    scene_name: str,
    device_ids: list,  # trainer
    num_coarse_samples: int,  # renderer
    num_fine_samples: int,
    weight_filtering_alpha: float,
    render_type: str,
    bounding_volume_size: float,
    num_gif_frames: int,
    spatial_encoding_l: int,  # encoder
    directional_encoding_l: int,
    nerf_type: str = "vanila",  # configs
    batch_size: int = 1024,
):
    # device
    if torch.cuda.is_available():
        torch.set_default_device(f"cuda:{device_ids[0]}")
        torch.cuda.set_device(device_ids[0])
    else:
        print("Cuda unavailable. Using CPU.")

    # dataset
    train_dataset = bldr.BlenderDataset(
        data_path,
        scene_name,
        "train",
    )

    # model
    if nerf_type == "vanila":
        coarse_network = net.NeRFNetwork(spatial_encoding_l, directional_encoding_l)
        fine_network = net.NeRFNetwork(spatial_encoding_l, directional_encoding_l)
        model = {"coarse_network": coarse_network, "fine_network": fine_network}
    elif nerf_type == "mip":
        network = net.NeRFNetwork(spatial_encoding_l, directional_encoding_l)
        model = {"network": network}
    elif nerf_type == "dreamfusion":
        raise NotImplementedError
    else:
        raise NotImplementedError

    # checkpoint
    _ = log.load_checkpoint(log_path, model)

    # dataparallel
    if len(device_ids) > 1:
        dp_model = {}
        for name, network in model.items():
            dp_model[name] = nn.DataParallel(network, device_ids)
        model = dp_model

    # intrinsic parameters
    focal_length = train_dataset.focal_length
    img_size = train_dataset.img_size
    pixel_density = train_dataset.pixel_density

    # horizontal movement
    print("Dynamic rendering: horizontal movement.")

    for i in tqdm(range(num_gif_frames // 2)):
        # sample cameras
        theta = 2 * np.pi * i / (num_gif_frames // 2)
        ellipse_a = 3 * bounding_volume_size
        ellipse_b = 2 * bounding_volume_size
        position = torch.tensor(
            [ellipse_a * np.cos(theta), ellipse_b * np.sin(theta), 0.0]
        ).to(torch.float)
        direction = -position / torch.linalg.norm(position)
        upvec = torch.tensor([0.0, 0, 1])
        pos_dir_up = {"position": position, "direction": direction, "upvec": upvec}
        near_dist = float(torch.linalg.norm(position) - bounding_volume_size)
        far_dist = float(torch.linalg.norm(position) + bounding_volume_size)
        camera = cam.Camera(
            pos_dir_up, focal_length, img_size, pixel_density, near_dist, far_dist
        )

        # render image
        with torch.no_grad():
            rendered_img = rend.render_image(
                model,  # input
                camera,
                num_coarse_samples,  # hyperparameters
                num_fine_samples,
                spatial_encoding_l,
                directional_encoding_l,
                weight_filtering_alpha,
                bounding_volume_size,
                nerf_type,  # conifgs
                render_type,
                batch_size,
            )

        # save image
        (log_path / "dynamic").mkdir(exist_ok=True)
        (log_path / "dynamic" / "images").mkdir(exist_ok=True)
        rendered_img = (255.0 * rendered_img).round().to(torch.uint8).cpu().numpy()
        iio.imwrite(
            log_path / "dynamic" / "images" / f"h{str(i).zfill(3)}.png",
            rendered_img,
        )

    # vertical movement
    print("Dynamic rendering: vertical movement.")

    for i in tqdm(range(num_gif_frames // 2)):
        # sample cameras
        theta = 2 * np.pi * i / (num_gif_frames // 2)
        ellipse_a = 3 * bounding_volume_size
        ellipse_c = 2 * bounding_volume_size
        position = torch.tensor(
            [ellipse_a * np.cos(theta), 0, ellipse_c * np.sin(theta)]
        ).to(torch.float)
        direction = -position / torch.linalg.norm(position)
        if position[2] != 0:
            upvec = (
                -position[2]
                / torch.abs(position[2])
                * torch.tensor([1.0, 0, -position[0] / position[2]])
            )
            upvec = upvec / torch.linalg.norm(upvec)
        else:
            upvec = torch.tensor([0.0, 0, position[0] / torch.abs(position[0])])
        pos_dir_up = {"position": position, "direction": direction, "upvec": upvec}
        near_dist = float(torch.linalg.norm(position) - bounding_volume_size)
        far_dist = float(torch.linalg.norm(position) + bounding_volume_size)
        camera = cam.Camera(
            pos_dir_up, focal_length, img_size, pixel_density, near_dist, far_dist
        )

        # render image
        with torch.no_grad():
            rendered_img = rend.render_image(
                model,  # input
                camera,
                num_coarse_samples,  # hyperparameters
                num_fine_samples,
                spatial_encoding_l,
                directional_encoding_l,
                weight_filtering_alpha,
                bounding_volume_size,
                nerf_type,  # conifgs
                render_type,
                batch_size,
            )

        # save image
        (log_path / "dynamic").mkdir(exist_ok=True)
        (log_path / "dynamic" / "images").mkdir(exist_ok=True)
        rendered_img = (255.0 * rendered_img).round().to(torch.uint8).cpu().numpy()
        iio.imwrite(
            log_path / "dynamic" / "images" / f"v{str(i).zfill(3)}.png",
            rendered_img,
        )

    # create gif
    imgs = []
    if num_gif_frames >= 100:
        gif_duration = 10.0
    else:
        gif_duration = 5.0
    gif_fps = round(num_gif_frames / gif_duration)
    for i in range(num_gif_frames // 2):
        imgs.append(
            iio.imread(log_path / "dynamic" / "images" / f"h{str(i).zfill(3)}.png")
        )
    for i in range(num_gif_frames // 2):
        imgs.append(
            iio.imread(log_path / "dynamic" / "images" / f"v{str(i).zfill(3)}.png")
        )
    iio.imwrite(
        log_path / "dynamic" / "dynamic.gif",
        imgs,
        plugin="pillow",
        duration=1000 / gif_fps,
        loop=0,
    )
