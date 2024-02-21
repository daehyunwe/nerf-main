import matplotlib.pyplot as plt
import torch


""" A world contains multiple cameras. """


class World:
    def __init__(self):
        self.cameras = []
        self.num_camera = 0

    def add_camera(self, camera):
        self.cameras.append(camera)
        self.num_camera = self.num_camera + 1

    def show_world(self, plot_line_length):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        alpha_scale = 1
        for camera in self.cameras:
            h, w = camera.img_size
            camera.plot_camera(
                ax,
                plot_line_length,
                torch.tensor([[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]]),
                alpha_scale=alpha_scale,
            )
            alpha_scale *= 0.7

        ax.set_aspect("equal")
        plt.savefig("world_fig")
        plt.show()
