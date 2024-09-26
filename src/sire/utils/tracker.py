import numpy as np
import torch

from trimesh.creation import icosphere


def cart2spher(cart_coords):
    # transform an Nx3 matrix of (unit length) Cartesian coordinates
    # into an Nx3 matrix with [r, phi, theta] spherical coordinates.
    # normalize cartesian coordinates
    if torch.is_tensor(cart_coords):
        cart_coords /= torch.linalg.norm(cart_coords, dim=1, keepdim=True)
        coords_spherical = torch.ones_like(cart_coords)
        theta = torch.arctan2(cart_coords[:, 1], cart_coords[:, 0])
        phi = torch.arccos(cart_coords[:, 2])
        theta += np.pi * 2 * (theta < 0).long()
    else:
        cart_coords = cart_coords / np.expand_dims(np.linalg.norm(cart_coords, axis=1), 1)
        coords_spherical = np.ones_like(cart_coords)
        theta = np.arctan2(cart_coords[:, 1], cart_coords[:, 0])
        phi = np.arccos(cart_coords[:, 2])
        theta += np.pi * 2 * (theta < 0).astype(int)

    coords_spherical[:, 1] = phi
    coords_spherical[:, 2] = theta
    return coords_spherical


class TrackerSphere(object):
    # this is the sphere object, that is used for obtaining the right data for the GEM-CNN
    # also contains the image/affine/centerline/annotations of the patient

    def __init__(self, subdivisions: int = 3):  # nverts = 642, 162, 42
        self.sphere = icosphere(subdivisions=subdivisions)
        self.sphereverts = cart2spher(self.sphere.vertices)
        self.cartverts = self.sphere.vertices

    def get_rays(self, npoints, ray_length, center=np.array([[0, 0, 0]])):
        """
        transform the Nx3 matrix containing the spherical coordinates of the vertices into image coordinates
        of all the points on the rays.

        Args:
            npoints: number of points on the ray
            ray_length:  real-world length of ray (in mms/cms)
            center: center of sphere, in world-coordinates

        Returns: (Nxraylength) x 3 matrix containing cartesian coordinates

        """
        rays = np.linspace(0, ray_length, npoints)
        sphereverts_long = self.sphereverts.repeat(npoints, axis=0)
        cart_coords = np.ones([self.sphereverts.shape[0] * npoints, 3])
        cart_coords[:, 0] = (
            np.tile(rays, self.sphereverts.shape[0]) * np.sin(sphereverts_long[:, 1]) * np.cos(sphereverts_long[:, 2])
        )
        cart_coords[:, 1] = (
            np.tile(rays, self.sphereverts.shape[0]) * np.sin(sphereverts_long[:, 1]) * np.sin(sphereverts_long[:, 2])
        )
        cart_coords[:, 2] = np.tile(rays, self.sphereverts.shape[0]) * np.cos(sphereverts_long[:, 1])
        return cart_coords + center  # [x, y, z] in world-coordinates

    def make_heatmap(self, directions, alpha, r):
        """
        make a discrete heatmap for phi,thetas given the objective tangent
        Args:
            directions: objective directions in spherical coordinates
            alpha: e^alpha*t (as in Sironi et al.)
            r: radius of Gaussian peak

        Returns: n_verts * 1 array, indicating objective directions as a Gaussian function on the sphere
        /!\ use 'continuous' sense of direction so true value of peak might differ!
        """
        heatmap = np.zeros([len(directions), self.sphereverts.shape[0]])
        for i, direction in enumerate(directions):
            great_circle_dist = np.arccos(
                np.sin(self.sphereverts[:, 1] - np.pi / 2) * np.sin(direction[1] - np.pi / 2)
                + np.cos(self.sphereverts[:, 1] - np.pi / 2)
                * np.cos(direction[1] - np.pi / 2)
                * np.cos(np.abs(self.sphereverts[:, 2] - direction[2]))
            )

            heatmap[i, :] = np.clip(
                (np.exp(alpha * (1 - great_circle_dist / r)) - 1) * (great_circle_dist < r).astype(int),
                0,
                np.exp(alpha),
            )
        return np.expand_dims(np.max(heatmap, axis=0), 1)
