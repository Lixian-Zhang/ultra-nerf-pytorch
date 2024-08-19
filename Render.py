import torch
import numpy as np
from torch.nn.functional import conv1d, conv2d

from Model import RenderParameter, NerualRadianceField
from Ray import Ray, RayBundle, pose_to_ray_bundle, pose_to_ray_bundle_linear
from utils import add_a_leading_zero, add_a_leading_one, plot_points, repeat_last_element

gaussian_kernal_1d = torch.tensor([0.2790,	0.4420,	0.2790]) # sigma = 0.5
gaussian_kernal_2d = gaussian_kernal_1d.reshape(-1, 1) * gaussian_kernal_1d.reshape(1, -1)
relaxation_temperature = torch.tensor([0.01])

def render_ray(ray: Ray, render_parameter: RenderParameter):
    # renders a single ray, uses 1d conv instead of 2d conv in the origional implementation
    distances_to_origin = ray.get_distances_to_origin()
    distances_between_points = add_a_leading_zero( torch.abs(distances_to_origin[:-1] - distances_to_origin[1:]) )
    attenuation_transmission = torch.exp( -torch.cumsum(distances_between_points * render_parameter.attenuation_coefficient, dim=0) )
    
    border_indicator = torch.bernoulli(render_parameter.border_probability)
    reflection_transmission = 1 - render_parameter.reflection_coefficient * border_indicator
    reflection_transmission = add_a_leading_one( torch.cumprod(reflection_transmission[:-1], dim=0) )
    
    border_indicator = border_indicator.unsqueeze(0).unsqueeze(0)
    kernel = gaussian_kernal_1d.unsqueeze(0).unsqueeze(0)
    border_convolution = conv1d(border_indicator, kernel, padding='same')
    border_convolution = border_convolution.squeeze()
    
    scatter_density = torch.bernoulli(render_parameter.scattering_density_coefficient)
    scatterers_map = scatter_density * render_parameter.scattering_amplitude
    scatterers_map = scatterers_map.unsqueeze(0).unsqueeze(0)
    psf_scatter = conv1d(scatterers_map, kernel, padding='same').squeeze()
    
    transmission = attenuation_transmission * reflection_transmission
    
    b = transmission * psf_scatter
    r = transmission * render_parameter.reflection_coefficient * border_convolution
    intensity_map = b + r
    return intensity_map

def render_ray_bundle(ray_bundle: RayBundle, nerf_model: NerualRadianceField):
    # renders a ray bundle 
    if ray_bundle.points is None or ray_bundle.distances_to_origin is None:
        raise ValueError('Please sample points before render.')

    render_parameter = RenderParameter(nerf_model(ray_bundle.points))
    distances_to_origin = ray_bundle.distances_to_origin

    distances_between_points = torch.abs(distances_to_origin[..., 1:, :] - distances_to_origin[..., :-1, :])
    distances_between_points = repeat_last_element(distances_between_points, dim=-2)
    attenuation = torch.exp(-render_parameter.attenuation_coefficient * distances_between_points)
    # attenuation_transmission = add_a_leading_one(torch.cumprod(attenuation[..., :-1, :], dim=-2), dim=-2)
    attenuation_transmission = torch.cumprod(attenuation, dim=-2)

    border_indicator = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(relaxation_temperature, probs=render_parameter.border_probability).sample()
    reflection_transmission = 1 - render_parameter.reflection_coefficient * border_indicator
    reflection_transmission = add_a_leading_one(torch.cumprod(reflection_transmission[..., :-1, :], dim=-2), dim=-2)
    border_indicator = border_indicator.unsqueeze(0).unsqueeze(0)
    kernel = gaussian_kernal_2d.unsqueeze(0).unsqueeze(0)
    border_convolution = conv2d(border_indicator, kernel, padding='same')
    border_convolution = border_convolution.squeeze()
    
    scatterers_density = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(relaxation_temperature, render_parameter.scattering_density_coefficient).sample()
    # scatterers_density = 0.5 * torch.ones_like(render_parameter.scattering_amplitude)
    scatterers_map = scatterers_density * render_parameter.scattering_amplitude
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