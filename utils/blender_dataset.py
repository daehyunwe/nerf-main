from pathlib import Path
from typing import Tuple
import json

import cv2
import numpy as np
import torch
import torch.utils.data as data


def _load_blender_dataset(
    scene_path: Path,
    data_type: str,
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    Loader for blender dataset.

    Input
        scene_path: path to the scene root, e.g. /models/nerf/data/lego/
        data_type: dataset type, test, train or val.
    Output
        images: size=[N, H, W, 3]
        poses: size=[N, 3, 4], camera extrinsic matrices.
        focal_length: focal length of the camera.
        img_size: size=[2], resolution [h, w] of the image.
        pixel_density: size=[2], number of pixels [m_x, m_y] per unit length.
    """
    with (scene_path / f"transforms_{data_type}.json").open() as file:
        poses_dict = json.load(file)

    images = []
    poses = []

    for frame in poses_dict["frames"]:
        img = cv2.imread((scene_path / f"{frame['file_path']}.png").as_posix())
        img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pose = torch.linalg.solve(
            torch.tensor(frame["transform_matrix"]), torch.eye(4)
        )[:3, :]
        # rotate 180 degrees along x-axis: graphics notation to vision notation
        pose[1] = -pose[1]
        pose[2] = -pose[2]

        images.append(img)
        poses.append(pose)

    images = torch.stack(images, dim=0) / 255.0
    poses = torch.stack(poses, dim=0)
    img_size = torch.tensor([images.size(dim=1), images.size(dim=2)])
    pixel_density = img_size // 2

    focal_length = (
        0.5
        * img_size[1]
        / np.tan(0.5 * poses_dict["camera_angle_x"])
        / pixel_density[0]
    )

    return images, poses, focal_length, img_size, pixel_density


class BlenderDataset(data.Dataset):
    """
    Synthetic blender dataset.
    """

    def __init__(
        self,
        data_path: Path,
        scene_name: str,
        data_type: str,
        near_dist: float,
        far_dist: float,
    ):
        super().__init__()

        self._scene_path = data_path / scene_name
        self._scene_name = scene_name
        self._data_type = data_type

        (
            self._images,
            self._poses,
            self.focal_length,
            self.img_size,
            self.pixel_density,
        ) = _load_blender_dataset(self._scene_path, self._data_type)

        self.near_dist = near_dist
        self.far_dist = far_dist

    def __len__(self):
        return self._images.size(dim=0)

    def __getitem__(self, index: int):
        image = self._images[index]
        pose = self._poses[index]
        return image, pose
