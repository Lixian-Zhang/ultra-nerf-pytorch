import torch
from torch.nn.functional import normalize

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
        self.direction = normalize(direction.reshape(1, 3), dim=1)          # the center ray direction
        self.plane_normal = normalize(plane_normal.reshape(1, 3), dim=1)    # normal vector of the plane where the rays inhabit

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


def get_ray_bundle(pose):
    origin = pose[:3, -1]
    rot_mat = pose[:3, :3]
    direction = rot_mat @ torch.Tensor([0, 0, 1]).reshape(-1, 1)
    plane_normal = rot_mat @ torch.Tensor([1, 0, 0]).reshape(-1, 1)
    central_angle = torch.pi / 2
    return RayBundle(origin, direction, plane_normal, central_angle)
   

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
    rb = RayBundle(o, torch.pi / 2, d, n)
    rds = rb._get_ray_directions(6)
    
    p, dis = rb.sample(0, 10, 2, 3)

    print(p.shape)
    print(p)
    print(dis.shape)
    print(dis)



if __name__ == '__main__':
    test_ray_bundle()