import torch

import src.graphics.rules as rls


class Camera:
    """
    A camera is specified by,
        pose: There are two options for representing the camera extrinsics.
        (1) Pos-dir-up dictionary
            position: size=[3], cartesian coordinate [x,y,z] of camera center.
            direction: size=[3], cartesian vector [x,y,z] of viewing direction.
            upvec: size=[3], cartesian vector [x,y,z] of the up vector, orthogonal to direction.
        (2) One-step transformation matrix
            transform_matrix: size=[3,4], world-to-camera extrinsic matrix.
        focal_length: focal length value.
        img_size: size=[2], image size [h,w] in pixel scale.
        pixel_density: size=[2], number of image pixels [m_x,m_y] per unit length.
        near_dist: distance to the near plane.
        far_dist: distance to the far plane.
    """

    def __init__(
        self,
        pose: dict | torch.Tensor,
        focal_length: float,
        img_size: torch.Tensor,
        pixel_density: torch.Tensor,
        near_dist: float,
        far_dist: float,
    ):
        if isinstance(pose, dict):
            device = pose["position"].device
        elif isinstance(pose, torch.Tensor):
            device = pose.device

        self.focal_length = focal_length
        self.img_size = img_size
        self.pixel_density = pixel_density
        self.near_dist = near_dist
        self.far_dist = far_dist
        self._intrinsic = torch.cat(  # [3,3]
            [
                torch.cat(
                    [
                        torch.diag(self.focal_length * self.pixel_density),
                        torch.reshape(torch.flip(self.img_size, dims=(0,)) / 2, [2, 1]),
                    ],
                    dim=1,
                ),
                torch.tensor([[0.0, 0, 1]], device=device),
            ],
            dim=0,
        )
        self._inverse_intrinsic = torch.linalg.solve(  # [3,3]
            self._intrinsic, torch.eye(3, device=device)
        )

        if isinstance(pose, dict):
            self.position = pose["position"]
            y_axis = -pose["upvec"] / torch.linalg.norm(pose["upvec"])
            z_axis = pose["direction"] / torch.linalg.norm(pose["direction"])
            x_axis = torch.tensor(
                [
                    y_axis[1] * z_axis[2] - y_axis[2] * z_axis[1],
                    -y_axis[0] * z_axis[2] + y_axis[2] * z_axis[0],
                    y_axis[0] * z_axis[1] - y_axis[1] * z_axis[0],
                ],
                device=device,
            )
            x_axis = x_axis / torch.linalg.norm(x_axis)
            self.rotation = torch.stack(
                [
                    x_axis,
                    y_axis,
                    z_axis,
                ],
                dim=0,
            )
            self._translation = -self.rotation @ self.position  # [3]
            self._extrinsic = torch.cat(  # [3,4], world-to-camera
                [self.rotation, torch.reshape(self._translation, [3, 1])], dim=1
            )
            self._inverse_extrinsic = torch.linalg.solve(  # [3,4], camera-to-world
                torch.cat(
                    [self._extrinsic, torch.tensor([[0.0, 0, 0, 1]], device=device)],
                    dim=0,
                ),
                torch.eye(4, device=device),
            )[:-1, :]

        elif isinstance(pose, torch.Tensor):
            self._extrinsic = pose  # [3,4], world-to-camera
            self.rotation = pose[:, :-1]  # [3,3]
            self._translation = pose[:, -1]  # [3]
            self.position = -self.rotation.T @ self._translation  # [3]
            self._inverse_extrinsic = torch.linalg.solve(  # [3,4], camera-to-world
                torch.cat(
                    [self._extrinsic, torch.tensor([[0.0, 0, 0, 1]], device=device)],
                    dim=0,
                ),
                torch.eye(4, device=device),
            )[:-1, :]

        else:
            raise AssertionError

    def world_points_to_image(self, points):
        """
        Transformation from world coordinate to image coordinate (pixel unit).
        Returns nan for points that do not reside in the image.

        Input
            points: torch.Tensor, size=[N,3]
        Output
            image_points: torch.Tensor, size=[N,2]
        """
        N = points.size(dim=0)

        image_points = rls.to_non_homogeneous(
            (self._intrinsic @ self._extrinsic @ rls.to_homogeneous(points).T).T
        )
        is_ge_zero = torch.all(image_points >= torch.zeros_like(image_points), dim=1)
        is_le_img_size = torch.logical_and(
            image_points[:, 0] <= self.img_size[1].expand(N),
            image_points[:, 1] <= self.img_size[0].expand(N),
        )
        image_points[torch.logical_and(is_ge_zero, is_le_img_size) == False] = float(
            "nan"
        )

        return image_points

    def image_points_to_pixel(self, points):
        """
        Returns the nearest image pixels given the image coordinates (pixel unit).
        Returns nan for points that do not reside in the image.

        Input
            points: torch.Tensor, size=[N,2]
        Output
            pixels: torch.Tensor, size=[N,2]
        """
        N = points.size(dim=0)

        pixels = torch.floor(points)
        is_ge_zero = torch.all(points >= torch.zeros_like(points), dim=1)
        is_le_img_size = torch.logical_and(
            points[:, 0] <= self.img_size[1].expand(N),
            points[:, 1] <= self.img_size[0].expand(N),
        )
        pixels[torch.logical_and(is_ge_zero, is_le_img_size) == False] = float("nan")

        return pixels

    def world_points_to_pixel(self, points):
        """
        One step function for direct conversion.

        Input
            points: torch.Tensor, size=[N,3]
        Output
            pixels: torch.Tensor, size=[N,2]
        """
        image_points = self.world_points_to_image(points)
        return self.image_points_to_pixel(image_points)

    def pixels_to_image(self, pixels):
        """
        Returns the image coordinates (pixel unit) of the given image pixels.
        Returns nan for pixels that do not reside in the image.

        Input
            pixels: torch.Tensor, size=[N,2]
        Output
            image_points: torch.Tensor, size=[N,2]
        """
        N = pixels.size(dim=0)

        image_points = pixels + 0.5
        is_ge_zero = torch.all(pixels >= torch.zeros_like(pixels), dim=1)
        is_le_img_size = torch.logical_and(
            pixels[:, 0] < self.img_size[1].expand(N),
            pixels[:, 1] < self.img_size[0].expand(N),
        )
        image_points[torch.logical_and(is_ge_zero, is_le_img_size) == False] = float(
            "nan"
        )

        return image_points

    def image_points_to_ray_directions(self, points):
        """
        Returns the direction of rays that start from camera center and direct to the image points.
        Returns nan for points that do not reside in the image.

        Input
            points: torch.Tensor, size=[N,2]
        Output
            directions: torch.Tensor, size=[N,3]
        """
        N = points.size(dim=0)

        directions = (
            self.rotation.T @ self._inverse_intrinsic @ rls.to_homogeneous(points).T
        ).T
        directions = directions / torch.linalg.norm(
            directions, dim=1, keepdim=True
        ).expand(-1, 3)
        is_ge_zero = torch.all(points >= torch.zeros_like(points), dim=1)
        is_le_img_size = torch.logical_and(
            points[:, 0] <= self.img_size[1].expand(N),
            points[:, 1] <= self.img_size[0].expand(N),
        )
        directions[torch.logical_and(is_ge_zero, is_le_img_size) == False] = float(
            "nan"
        )

        return directions

    def pixels_to_ray_directions(self, pixels):
        """
        Returns the direction of rays that start from camera center and direct to the pixels.

        Input
            pixels: torch.Tensor, size=[N,2]
        Output
            directions: torch.Tensor, size=[N,3]
        """
        image_points = self.pixels_to_image(pixels)
        return self.image_points_to_ray_directions(image_points)
