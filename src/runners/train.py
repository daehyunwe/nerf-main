from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import imageio.v3 as iio

import src.graphics.camera as cam
import src.network.network as net
import src.renderer.renderer as rend
import utils.blender_dataset as bldr
import utils.logger as log


def _sample_pixel_coordinates(
    batch_size: int,
    img_size: torch.Tensor,
) -> torch.Tensor:
    """
    Sample pixel indices.

    Input
        batch_size: number of indices. May be reduced if too large.
        img_size: size=[2], image resolution [h, w]

    Output
        pixel_coordinates: size=[N, 2].
    """
    flat_coordinates = torch.tensor(
        np.random.choice(
            img_size[0].item() * img_size[1].item(),
            size=batch_size,
            replace=False,
        )
    )
    pixel_coordinates = torch.cat(
        [
            (flat_coordinates // img_size[1]).view(batch_size, 1),
            (flat_coordinates % img_size[1]).view(batch_size, 1),
        ],
        dim=1,
    )

    return pixel_coordinates


def _train_one_epoch(
    epoch: int,  # input
    max_iter: int,
    train_dataset: bldr.BlenderDataset,
    train_loader: data.DataLoader,
    model: dict,
    optimizer: optim.Optimizer,
    scheduler,
    batch_size: int,  # trainer hyperparameters
    num_cameras_each_iter: int,
    num_coarse_samples: int,  # renderer hyperparameters
    num_fine_samples: int,
    spatial_encoding_l: int,
    directional_encoding_l: int,
    bounding_volume_size: float,
    weight_filtering_alpha: float,
    coarse_fine_loss_lambda: float,
    nerf_type: str = "vanila",  # configs
    render_type: str = "albedo",
) -> float:
    avg_loss = 0.0

    cameras = []
    imgs = []

    for i, batch in enumerate(train_loader):
        img, pose = batch
        img = img.squeeze()
        pose = pose.squeeze()

        camera = cam.Camera(
            pose,
            train_dataset.focal_length,
            train_dataset.img_size,
            train_dataset.pixel_density,
            train_dataset.near_dist,
            train_dataset.far_dist,
        )
        cameras.append(camera)
        imgs.append(img)

    for i in range(len(train_dataset)):
        optimizer.zero_grad()

        ray_origins = []
        ray_directions = []
        gt_pixel_colors = []
        camera_indices = np.random.choice(
            len(train_dataset), num_cameras_each_iter, replace=False
        )

        for j in camera_indices:
            pixel_coordinates = _sample_pixel_coordinates(
                batch_size // len(camera_indices), train_dataset.img_size
            )
            N = pixel_coordinates.size(dim=0)
            ray_origins.append(cameras[j].position.view(1, 3).expand(N, -1))
            ray_directions.append(
                cameras[j].pixels_to_ray_directions(pixel_coordinates)
            )
            gt_pixel_colors.append(
                imgs[j].view(-1, 3)[
                    train_dataset.img_size[1] * pixel_coordinates[:, 1]
                    + pixel_coordinates[:, 0]
                ]
            )

        ray_origins = torch.cat(ray_origins, dim=0)
        ray_directions = torch.cat(ray_directions, dim=0)
        gt_pixel_colors = torch.cat(gt_pixel_colors, dim=0)

        coarse_color, fine_color = rend.render_pixels(
            model,
            ray_origins,
            ray_directions,
            train_dataset.pixel_density,
            num_coarse_samples,
            num_fine_samples,
            spatial_encoding_l,
            directional_encoding_l,
            camera.near_dist,
            camera.far_dist,
            bounding_volume_size,
            weight_filtering_alpha,
            nerf_type,
            render_type,
        )

        loss_func = nn.MSELoss()
        coarse_loss = loss_func(
            gt_pixel_colors,
            coarse_color,
        )
        fine_loss = loss_func(
            gt_pixel_colors,
            fine_color,
        )
        loss = coarse_fine_loss_lambda * coarse_loss + fine_loss

        iter_msg = f"[{i+1} / {len(train_dataset)}] "
        epoch_msg = f"[{epoch+1} / {max_iter // len(train_dataset)}] "
        msg = "Training loss " + iter_msg + epoch_msg + f": {loss.item()}"
        print(" " * len(msg), end="")
        print("\r", end="")
        print(msg, end="")

        avg_loss = avg_loss + loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss /= len(train_dataset)

    return avg_loss


def _validate_one_epoch(
    epoch: int,  # input
    max_iter: int,
    log_path: Path,
    val_dataset: bldr.BlenderDataset,
    val_loader: data.DataLoader,
    model: dict,
    num_coarse_samples: int,  # renderer hyperparameters
    num_fine_samples: int,
    spatial_encoding_l: int,
    directional_encoding_l: int,
    bounding_volume_size: float,
    weight_filtering_alpha: float,
    nerf_type: str,  # configs
    render_type: str,
    save_images: bool,
    batch_size: int = 1024,
) -> float:
    avg_loss = 0.0

    for i, batch in enumerate(val_loader):
        if i >= 10:
            break

        img, pose = batch
        img = img.squeeze()
        pose = pose.squeeze()

        camera = cam.Camera(
            pose,
            val_dataset.focal_length,
            val_dataset.img_size,
            val_dataset.pixel_density,
            val_dataset.near_dist,
            val_dataset.far_dist,
        )

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

            # compute loss
            loss_func = nn.MSELoss()
            loss = loss_func(
                img.view(-1, 3),
                rendered_img.view(-1, 3),
            )

            iter_msg = f"[{i+1} / {10}] "
            msg = "Validation loss " + iter_msg + f": {loss.item()}"
            print(" " * len(msg), end="")
            print("\r", end="")
            print(msg, end="")

            avg_loss = avg_loss + loss.item()

            # save image
            if save_images:
                (log_path / f"{str(epoch).zfill(6)}").mkdir(exist_ok=True)
                (log_path / "val_originals").mkdir(exist_ok=True)
                rendered_img = (
                    (255.0 * rendered_img).round().to(torch.uint8).cpu().numpy()
                )
                img = (255.0 * img).round().to(torch.uint8).cpu().numpy()

                iio.imwrite(
                    log_path / f"{str(epoch).zfill(6)}" / f"{str(i).zfill(6)}.png",
                    rendered_img,
                )
                iio.imwrite(
                    log_path / "val_originals" / f"{str(i).zfill(6)}.png",
                    img,
                )

    avg_loss /= 10

    return avg_loss


def train(
    log_path: Path,  # path
    data_path: Path,
    scene_name: str,
    device_ids: list,  # trainer
    batch_size: int,
    max_iter: int,
    initial_lr: float,
    final_lr: float,
    coarse_fine_loss_lambda: float,
    validate_for_every: int,
    save_ckpt_for_every: int,
    num_cameras_each_iter: int,
    num_coarse_samples: int,  # renderer
    num_fine_samples: int,
    weight_filtering_alpha: float,
    render_type: str,
    near_dist: float,
    far_dist: float,
    bounding_volume_size: float,
    spatial_encoding_l: int,  # encoder
    directional_encoding_l: int,
    nerf_type: str = "vanila",  # configs
):
    # device
    if torch.cuda.is_available():
        torch.set_default_device(f"cuda:{device_ids[0]}")
        torch.cuda.set_device(device_ids[0])
    else:
        print("Cuda unavailable. Using CPU.")

    # dataset
    train_dataset = bldr.BlenderDataset(
        data_path, scene_name, "train", near_dist, far_dist
    )
    train_loader = data.DataLoader(train_dataset)
    val_dataset = bldr.BlenderDataset(data_path, scene_name, "val", near_dist, far_dist)
    val_loader = data.DataLoader(val_dataset)

    # model, optimizer, and scheduler
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
    params = []
    for _, network in model.items():
        params += list(network.parameters())
    optimizer = optim.Adam(params, initial_lr)
    gamma = pow(final_lr / initial_lr, 1 / max_iter)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    # checkpoint
    start_epoch = log.load_checkpoint(log_path, model, optimizer, scheduler)

    # dataparallel
    if len(device_ids) > 1:
        dp_model = {}
        for name, network in model.items():
            dp_model[name] = nn.DataParallel(network, device_ids)
        model = dp_model

    # iteration
    for epoch in range(start_epoch, max_iter // len(train_dataset)):
        # train
        train_loss = _train_one_epoch(
            epoch,  # input
            max_iter,
            train_dataset,
            train_loader,
            model,
            optimizer,
            scheduler,
            batch_size,  # trainer hyperparameters
            num_cameras_each_iter,
            num_coarse_samples,  # renderer hyperparameters
            num_fine_samples,
            spatial_encoding_l,
            directional_encoding_l,
            bounding_volume_size,
            weight_filtering_alpha,
            coarse_fine_loss_lambda,
            nerf_type=nerf_type,  # configs
            render_type=render_type,
        )

        # validate
        if (epoch + 1) % validate_for_every == 0:
            val_loss = _validate_one_epoch(
                epoch,  # input
                max_iter,
                log_path,
                val_dataset,
                val_loader,
                model,
                num_coarse_samples,  # renderer hyperparameters
                num_fine_samples,
                spatial_encoding_l,
                directional_encoding_l,
                bounding_volume_size,
                weight_filtering_alpha,
                nerf_type=nerf_type,  # configs
                render_type=render_type,
                save_images=True,
                batch_size=batch_size,
            )
        else:
            val_loss = None

        # save checkpoint
        if (epoch + 1) % save_ckpt_for_every == 0:
            log.save_checkpoint(log_path, epoch, model, optimizer, scheduler)

        # save loss
        log.save_loss(log_path, epoch, train_loss, val_loss)
