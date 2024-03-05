from pathlib import Path
import json
import shutil

import cv2
import numpy as np


def down_sample(scene_path: Path, down_size: list):
    """
    Down sample blender dataset images and store them into scene_path.

    Input
        scene_path: path to the scene root, e.g. /models/nerf/data/lego/
        down_size: downsampled image resolution [h, w].
    """
    scene_name = scene_path.name
    down_path = scene_path.parent / f"{scene_name}_down"

    # downsample and store images into down_path
    down_path.mkdir(exist_ok=True)
    for data_type in ["test", "train", "val"]:
        (down_path / f"{data_type}").mkdir(exist_ok=True)

        # json
        with (scene_path / f"transforms_{data_type}.json").open() as file:
            poses_dict = json.load(file)
        shutil.copy(
            scene_path / f"transforms_{data_type}.json",
            down_path / f"transforms_{data_type}.json",
        )

        # images
        for frame in poses_dict["frames"]:
            img = cv2.imread(
                (scene_path / f"{frame['file_path']}.png").as_posix(),
                cv2.IMREAD_UNCHANGED,
            )
            down_img = cv2.resize(img, dsize=tuple(down_size))
            cv2.imwrite((down_path / f"{frame['file_path']}.png").as_posix(), down_img)
