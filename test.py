import torch
import numpy as np
from tqdm import tqdm, trange
import imageio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
torch.autograd.set_detect_anomaly(True)

from pytorch_msssim import SSIM

from Ray import pose_to_ray_bundle_linear
from Model import NerualRadianceField
from Render import render_ray_bundle

images = np.load('./data/images.npy')
poses = np.load('./data/poses.npy')
images = images.astype(np.float32) / 255
images, poses = images[:100], poses[:100]

nerf = NerualRadianceField()
nerf.load_state_dict(torch.load('nerf_100.pt'))
i = 0
for image, pose in tqdm(zip(images, poses), total=len(poses)):
    image_pt, pose = torch.from_numpy(image).to(device), torch.from_numpy(pose).to(device)
    rb = pose_to_ray_bundle_linear(pose)
    rb.sample(0, 140 * 0.001, 80 * 0.001, image.shape[0], image.shape[1])
    rendered_image = render_ray_bundle(rb, nerf)
    side_by_side_image = np.concatenate([ (image * 255).astype(np.uint8), (rendered_image.numpy(force=True) * 255).astype(np.uint8) ], axis=-1)
    imageio.imwrite(f'./test_result/{i}.png', side_by_side_image)
    i += 1

    

    
    





    