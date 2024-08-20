import torch
from torch.nn.functional import normalize
from utils import plot_points
class Ray:

    def __init__(self, origin, direction):
        self.origin = origin                   # a 3D point
        self.direction = normalize(direction, dim=0)  # a unit normal vector
        self.points = None
        self.distances_to_origin = None

    def get_points(self, near=None, far=None, number_of_points=0):
        # output is tensor of shape (number_of_points, 3)
        if near is None or far is None or number_of_points == 0:
            # use cached
            if self.points is None:
                raise ValueError('get_point must be called once with arguments specified before calling with default arguments.')
            return self.points
        # uniform sampling
        distances_to_origin = torch.linspace(near, far, number_of_points)
        points = self.origin.reshape(1, 3) + distances_to_origin.reshape(-1, 1) * self.direction.reshape(1, 3)
        self.points = points
        self.distances_to_origin = distances_to_origin
        return self.points

    def get_distances_to_origin(self):
        if self.points is None:
            raise ValueError('get_point must be called once before calling get_distances_to_origin')
        assert self.distances_to_origin is not None
        return self.distances_to_origin


class RayBundle:

    def __init__(self, origin, direction, plane_normal, central_angle):
        # Circular sector shaped ray bundle
        self.origin = origin.reshape(1, 3)                # origin of the circle
        self.central_angle = min(abs(central_angle), torch.pi)              # central angle in rad within [0, pi], defines how wide rays spread
        self.direction = normalize(direction.reshape(1, 3), dim=1)          # the center ray direction direction must be perpendicular to plane_normal 
        self.plane_normal = normalize(plane_normal.reshape(1, 3), dim=1)    # normal vector of the plane where the rays inhabit
        self.points = None
        self.distances_to_origin = None

    def sample(self, near, far, num_points_per_ray, num_rays):
        # sample points in a fan-shape area
        # near: distance between top most pixel to origin
        # far:  distance between bottom most pixel to origin
        # num_rays: width
        # num_points_per_ray: height
        distances_to_origin = torch.linspace(near, far, num_points_per_ray)
        points = self.origin.reshape(1, 1, 3) + distances_to_origin.reshape(-1, 1, 1) * self._get_ray_directions(num_rays).reshape(1, -1, 3)
        self.points = points
        self.distances_to_origin = distances_to_origin.unsqueeze(-1).broadcast_to(points.shape[:2])
        return self.points, self.distances_to_origin

    def _get_ray_directions(self, num_rays):
        ray_angles = torch.linspace(-self.central_angle / 2, self.central_angle / 2, num_rays).reshape(1, -1)
        alphas = ray_angles[:, -(num_rays // 2):]
        sin_alphas = torch.sin(alphas).reshape(-1, 1)
        cos_alphas = torch.cos(alphas).reshape(-1, 1)
        cross = torch.linalg.cross(self.direction, self.plane_normal)
        dirs1 = sin_alphas * cross + cos_alphas * self.direction
        dirs2 = sin_alphas * -cross + cos_alphas * self.direction
        if num_rays & 1 == 0: # even
            directions = torch.concat([dirs1.flip([0]), dirs2], 0)
        else: # odd
            directions = torch.concat([dirs1.flip([0]), self.direction, dirs2], 0)
        return normalize(directions, dim=1)

class RayBundleLinear:

    def __init__(self, origin, direction, plane_normal):
        # Circular sector shaped ray bundle
        self.origin = origin.reshape(1, 3)                # origin of 
        self.direction = normalize(direction.reshape(1, 3), dim=1)          # the center ray direction direction must be perpendicular to plane_normal 
        self.plane_normal = normalize(plane_normal.reshape(1, 3), dim=1)    # normal vector of the plane where the rays inhabit
        if not torch.allclose(torch.dot(self.direction.flatten(), self.plane_normal.flatten()), torch.zeros(1), atol=5e-2):
            raise ValueError(f'Direction vector and plane normal vector must be perpendicular to each other, got {self.direction} and {self.plane_normal} whose dot product is {torch.dot(self.direction.flatten(), self.plane_normal.flatten())}')
        self.points = None
        self.distances_to_origin = None

    def sample(self, near, far, width, num_points_per_ray, num_rays):
        # sample points in a fan-shape area
        # near: distance between top most pixel to origin
        # far:  distance between bottom most pixel to origin
        # width: distance between left most ray and right most ray
        # num_rays: image width
        # num_points_per_ray: image height
        distances_to_origin = torch.linspace(near, far, num_points_per_ray)
        points = self._get_ray_origins(num_rays, width).reshape(1, -1, 3) + (distances_to_origin.reshape(-1, 1) * self.direction.reshape(1, 3)).reshape(-1, 1, 3)
        self.points = points
        self.distances_to_origin = distances_to_origin.unsqueeze(-1).broadcast_to(points.shape[:2])
        return self.points, self.distances_to_origin

    def _get_ray_origins(self, num_rays, width):
        distances = torch.linspace(-width / 2, width / 2, num_rays).reshape(-1, 1)
        line_of_origins = torch.linalg.cross(self.direction, self.plane_normal).reshape(1, 3)
        # print(self.origin.device) c
        # print(distances.device) g
        # print(line_of_origins.device) c
        return self.origin + distances * line_of_origins

def pose_to_ray_bundle(pose):
    origin = pose[:3, -1]
    rot_mat = pose[:3, :3]
    direction = rot_mat @ torch.tensor([0, 0, 1]).reshape(-1, 1)
    plane_normal = rot_mat @ torch.tensor([1, 0, 0]).reshape(-1, 1)
    central_angle = torch.pi / 2
    return RayBundle(origin, direction, plane_normal, central_angle)
   
def pose_to_ray_bundle_linear(pose, offset=torch.eye(4)):
    origin = pose[:3, -1] + offset[:3, -1]
    rot_mat = pose[:3, :3] @ offset[:3, :3]
    d = torch.linalg.det(rot_mat)
    if not torch.allclose(d, torch.ones_like(d), rtol=0.001):
        raise ValueError(f'Invalid pose, determinant of the rotation matrix is {d}.')
    direction = rot_mat @ torch.tensor([0, 0, 1], dtype=torch.float).reshape(-1, 1)
    plane_normal = rot_mat @ torch.tensor([1, 0, 0], dtype=torch.float).reshape(-1, 1)
    return RayBundleLinear(origin, direction, plane_normal)

def test_ray():
    o = torch.Tensor([1, 1, 0])
    d = torch.Tensor([0, 3, 4])
    r = Ray(o, d)
    points = r.get_points(0, 10, 11)
    dists = r.get_distances_to_origin()
    print(points)
    print(dists)

def test_ray_bundle():
    o = torch.Tensor([0, 0, 0])
    d = torch.Tensor([1, 1, 0])
    n = torch.Tensor([0, 0, 1])
    rb = RayBundle(o, d, n, torch.pi / 2)
    
    p, dis = rb.sample(1, 10, 4, 5)

    print(p.shape)
    print(p)

    dis2 = torch.linalg.norm(p - rb.origin.reshape(1, 1, 3), dim=-1)
    assert torch.allclose(dis, dis2)


def test_ray_bundle_linear():
    pose = torch.eye(4)
    rbl = pose_to_ray_bundle_linear(pose)
    ref, _ = rbl.sample(0, 5, 3, 5, 3)
    pose[:3, :3] = torch.Tensor([
        [0.7071068,  0.0000000,  0.7071068],
        [0.0000000,  1.0000000,  0.0000000],
        [-0.7071068,  0.0000000,  0.7071068],
    ])
    pose[:3, -1] = torch.Tensor([-1, 0, 0])
    rbl2 = pose_to_ray_bundle_linear(pose)
    points, _ = rbl2.sample(0, 5, 3, 5, 3)
    plot_points(points, ref)


 
if __name__ == '__main__':
    test_ray_bundle_linear()