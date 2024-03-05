from pathlib import Path
import argparse
import yaml

import src.runners.dynamic_render as render
import src.runners.train as train
import utils.down_sampler as down


def main():
    # resolve path
    root_path = Path(__file__).parent

    # load configs
    with (root_path / "configs" / "nerf.yaml").open() as file:
        nerf_config = yaml.safe_load(file)
    nerf_type = nerf_config["nerf_type"]
    common_config = nerf_config["common"]
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

    if not common_config["dataset"]["down_size"] is None:
        if not (
            root_path / "data" / f"{common_config['dataset']['scene_name']}_down"
        ).exists():
            down.down_sample(
                root_path / "data" / common_config["dataset"]["scene_name"],
                common_config["dataset"]["down_size"],
            )
        scene_name = f"{common_config['dataset']['scene_name']}_down"
    else:
        scene_name = common_config["dataset"]["scene_name"]

    if args.train:
        train.train(
            root_path / "log",  # path
            root_path / "data",
            scene_name,
            common_config["trainer"]["device_ids"],  # trainer
            common_config["trainer"]["batch_size"],
            common_config["trainer"]["max_iter"],
            nerf_config["trainer"]["initial_lr"],
            nerf_config["trainer"]["final_lr"],
            nerf_config["trainer"]["coarse_fine_loss_lambda"],
            common_config["trainer"]["validate_for_every"],
            common_config["trainer"]["save_ckpt_for_every"],
            common_config["trainer"]["num_cameras_each_iter"],
            nerf_config["renderer"]["num_coarse_samples"],  # renderer
            nerf_config["renderer"]["num_fine_samples"],
            nerf_config["renderer"]["weight_filtering_alpha"],
            nerf_config["renderer"]["render_type"],
            common_config["renderer"]["near_dist"],
            common_config["renderer"]["far_dist"],
            common_config["renderer"]["bounding_volume_size"],
            nerf_config["encoder"]["spatial_encoding_l"],  # encoder
            nerf_config["encoder"]["directional_encoding_l"],
            nerf_type=nerf_type,  # configs
        )

    elif args.render:
        render.dynamic_render(
            root_path / "log",
            root_path / "data",
            scene_name,
            common_config["trainer"]["device_ids"],
            nerf_config["renderer"]["num_coarse_samples"],
            nerf_config["renderer"]["num_fine_samples"],
            nerf_config["renderer"]["weight_filtering_alpha"],
            nerf_config["renderer"]["render_type"],
            common_config["renderer"]["bounding_volume_size"],
            common_config["renderer"]["num_gif_frames"],
            nerf_config["encoder"]["spatial_encoding_l"],
            nerf_config["encoder"]["directional_encoding_l"],
            nerf_type=nerf_type,
            batch_size=common_config["trainer"]["batch_size"],
        )

    else:
        raise AssertionError


if __name__ == "__main__":
    main()
