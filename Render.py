import torch
from torch.nn.functional import conv2d

from Model import RenderParameter, NerualRadianceField
from Ray import RayBundle, pose_to_ray_bundle_linear
from utils import add_a_leading_one, repeat_last_element, sample_bernoulli

gaussian_kernal_1d = torch.tensor([0.2790,	0.4420,	0.2790]) # sigma = 0.5
gaussian_kernal_2d = gaussian_kernal_1d.reshape(-1, 1) * gaussian_kernal_1d.reshape(1, -1)

def render_ray_bundle(ray_bundle: RayBundle, nerf_model: NerualRadianceField):
    # renders a ray bundle 
    if ray_bundle.points is None or ray_bundle.distances_to_origin is None:
        raise ValueError('Please sample points before render.')

    render_parameter = RenderParameter(nerf_model(ray_bundle.points))
    distances_to_origin = ray_bundle.distances_to_origin

    distances_between_points = torch.abs(distances_to_origin[..., 1:, :] - distances_to_origin[..., :-1, :])
    distances_between_points = repeat_last_element(distances_between_points, dim=-2)
    attenuation = torch.exp(-render_parameter.attenuation_coefficient * distances_between_points)
    attenuation_transmission = torch.cumprod(attenuation, dim=-2)

    border_indicator = sample_bernoulli(render_parameter.border_probability)
    reflection_transmission = 1 - render_parameter.reflection_coefficient * border_indicator
    reflection_transmission = add_a_leading_one(torch.cumprod(reflection_transmission[..., :-1, :], dim=-2), dim=-2)
    border_indicator = border_indicator.unsqueeze(0).unsqueeze(0)
    kernel = gaussian_kernal_2d.unsqueeze(0).unsqueeze(0)
    border_convolution = conv2d(border_indicator, kernel, padding='same')
    border_convolution = border_convolution.squeeze()
    
    scatterers = sample_bernoulli(render_parameter.scattering_density_coefficient)
    scatterers_map = scatterers * render_parameter.scattering_amplitude
    scatterers_map = scatterers_map.unsqueeze(0).unsqueeze(0)
    psf_scatter = conv2d(scatterers_map, kernel, padding='same').squeeze()

    transmission = attenuation_transmission * reflection_transmission
    b = transmission * psf_scatter
    r = transmission * render_parameter.reflection_coefficient * border_convolution
    intensity_map = b + r

    return intensity_map

def test():
    nerf = NerualRadianceField()
    pose = torch.eye(4)
    pose[:3, :3] = torch.Tensor([
        [  0.2760942, -0.5594423,  0.7815346],
        [  0.9069170,  0.4208754, -0.0191151],
        [ -0.3182349,  0.7140646,  0.6235690],
    ])
    pose.requires_grad_(True)
    rb = pose_to_ray_bundle_linear(pose)
    rb.sample(0.04, 0.16, 0.5, 512, 256)
    i = render_ray_bundle(rb, nerf)





if __name__ == '__main__':
    test()