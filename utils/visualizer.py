import numpy as np
import torch

"""
How to use.

# Import
import matplotlib.pyplot as plt
import numpy as np
import torch

import .utils.visualizer as vis

# Initialize
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot sphere
vis.plot_sphere(ax, 1)

# Plot cube
vis.plot_cube(ax, 2)

# Plot square
vis.plot_square(ax, np.eye(3), np.array([1,1,1]), 3)

# Plot lines
vis.plot_lines(ax, np.array([[0.,0,0],[0,0,0]]), torch.tensor([[1.,1,1],[1,1,-1]]), np.array([5,3]))

# Plot points
vis.plot_points(ax, torch.tensor([[0,0,0],[1,1,1],[2,2,2]]))

# Decorate
ax.set_aspect("equal")

# Show
plt.savefig("fig")
plt.show()
"""


def plot_sphere(ax, radius, color="b", alpha=0.1):
    if torch.is_tensor(radius):
        radius = radius.numpy(force=True)
    azimuth = np.linspace(0, 2 * np.pi, 30)
    elevation = np.linspace(-np.pi / 2, np.pi / 2, 30)
    azimuth, elevation = np.meshgrid(azimuth, elevation)
    distance = radius * np.ones_like(azimuth)

    X = distance * np.cos(elevation) * np.cos(azimuth)
    Y = distance * np.cos(elevation) * np.sin(azimuth)
    Z = distance * np.sin(elevation)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)


def plot_cube(ax, length, color="g", alpha=0.1):
    if torch.is_tensor(length):
        length = length.numpy(force=True)
    x = np.linspace(-length / 2, length / 2, 2)
    y = np.linspace(-length / 2, length / 2, 2)
    x, y = np.meshgrid(x, y)
    c = length / 2 * np.ones_like(x)

    X = np.stack([c, -c, x, x, x, x])
    Y = np.stack([x, x, c, -c, y, y])
    Z = np.stack([y, y, y, y, c, -c])
    for i in range(X.shape[0]):
        ax.plot_surface(X[i], Y[i], Z[i], color=color, alpha=alpha)


def plot_square(ax, rotation, translation, length, color="r", alpha=0.2):
    if torch.is_tensor(rotation):
        rotation = rotation.numpy(force=True)
    if torch.is_tensor(translation):
        translation = translation.numpy(force=True)
    if torch.is_tensor(length):
        length = length.numpy(force=True)

    x = np.linspace(-length / 2, length / 2, 10)
    y = np.linspace(-length / 2, length / 2, 10)
    x, y = np.meshgrid(x, y)
    grid_shape = x.shape
    x, y = x.flatten().reshape([-1, 1]), y.flatten().reshape([-1, 1])
    points = np.concatenate((x, y, np.zeros_like(x)), axis=1)
    trans_points = (
        rotation.T
        @ (points - np.tile(translation.reshape([1, -1]), (points.shape[0], 1))).T
    ).T
    X = trans_points[:, 0].reshape(grid_shape)
    Y = trans_points[:, 1].reshape(grid_shape)
    Z = trans_points[:, 2].reshape(grid_shape)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)


def plot_lines(ax, origins, directions, lengths, color="y", alpha=0.3):
    if torch.is_tensor(origins):
        origins = origins.numpy(force=True)
    if torch.is_tensor(directions):
        directions = directions.numpy(force=True)
    if torch.is_tensor(lengths):
        lengths = lengths.numpy(force=True)

    batch_size = origins.shape[0]
    directions = directions / np.tile(
        np.linalg.norm(directions, axis=1).reshape([batch_size, 1]), (1, 3)
    )
    t = np.linspace(np.zeros_like(lengths), lengths, 2)

    for i in range(batch_size):
        X = np.full_like(t[:, i], origins[i, 0]) + t[:, i] * directions[i, 0]
        Y = np.full_like(t[:, i], origins[i, 1]) + t[:, i] * directions[i, 1]
        Z = np.full_like(t[:, i], origins[i, 2]) + t[:, i] * directions[i, 2]
        ax.plot(X, Y, Z, color=color, alpha=alpha)


def plot_points(ax, points, color="r", alpha=0.5):
    if torch.is_tensor(points):
        points = points.numpy(force=True)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=alpha)
