from pathlib import Path
import argparse
import yaml

import src.runners.train as train
import src.runners.dynamic_render as render


def main():
    # resolve path
    ROOT_PATH = Path(__file__).parent

    # load configs
    with (ROOT_PATH / "configs" / "nerf.yaml").open() as file:
        nerf_config = yaml.safe_load(file)
    nerf_type = nerf_config["nerf_type"]
    nerf_config = nerf_config[nerf_type]

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--train", help="initiate nerf training", action="store_true"
    )
    parser.add_argument(
        "-r",
        "--render",
        help="generate gif with dynamic rendering",
        action="store_true",
    )
    args = parser.parse_args()

    if args.train:
        train.train(
            ROOT_PATH / "log",  # path
            ROOT_PATH / "data",
            nerf_config["dataset"]["scene_name"],
            nerf_config["trainer"]["device_ids"],  # trainer
            nerf_config["trainer"]["batch_size"],
            nerf_config["trainer"]["max_iter"],
            nerf_config["trainer"]["initial_lr"],
            nerf_config["trainer"]["final_lr"],
            nerf_config["trainer"]["coarse_fine_loss_lambda"],
            nerf_config["trainer"]["validate_for_every"],
            nerf_config["trainer"]["save_ckpt_for_every"],
            nerf_config["trainer"]["num_cameras_each_iter"],
            nerf_config["renderer"]["num_coarse_samples"],  # renderer
            nerf_config["renderer"]["num_fine_samples"],
            nerf_config["renderer"]["weight_filtering_alpha"],
            nerf_config["renderer"]["render_type"],
            nerf_config["renderer"]["near_dist"],
            nerf_config["renderer"]["far_dist"],
            nerf_config["renderer"]["bounding_volume_size"],
            nerf_config["encoder"]["spatial_encoding_l"],  # encoder
            nerf_config["encoder"]["directional_encoding_l"],
            nerf_type=nerf_type,  # configs
        )

    elif args.render:
        render.dynamic_render(
            ROOT_PATH / "log",
            ROOT_PATH / "data",
            nerf_config["dataset"]["scene_name"],
            nerf_config["trainer"]["device_ids"],
            nerf_config["renderer"]["num_coarse_samples"],
            nerf_config["renderer"]["num_fine_samples"],
            nerf_config["renderer"]["weight_filtering_alpha"],
            nerf_config["renderer"]["render_type"],
            nerf_config["renderer"]["bounding_volume_size"],
            nerf_config["renderer"]["num_gif_frames"],
            nerf_config["encoder"]["spatial_encoding_l"],
            nerf_config["encoder"]["directional_encoding_l"],
            nerf_type=nerf_type,
            batch_size=nerf_config["trainer"]["batch_size"],
        )

    else:
        raise AssertionError


if __name__ == "__main__":
    main()
