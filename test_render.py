import torch
from torch.optim import lr_scheduler
import numpy as np
from numpy.random import Generator
from tqdm import tqdm, trange
import wandb
import imageio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
torch.autograd.set_detect_anomaly(True)

from pytorch_msssim import SSIM, MS_SSIM

from Ray import pose_to_ray_bundle_linear
from Model import NerualRadianceField
from a import Test
from Render import render_ray_bundle

# run = wandb.init(project='ultra-nerf-pytorch')

# images = np.load('./data/images.npy')
# poses = np.load('./data/poses.npy')
images = np.load('.\data\images.npy')
poses = np.load('.\data\poses.npy')
images = images.astype(np.float32) / 255
images, poses = images[:1], poses[:1]
offset = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float)

ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=1)
l2 = torch.nn.MSELoss(reduction='mean')

nerf = Test((512, 256, 5))
start_lr = 1e-2
end_lr = 1e-5
epochs = 2000
adam = torch.optim.Adam(nerf.parameters(), lr=start_lr)
scheduler = lr_scheduler.ExponentialLR(adam, gamma=pow(end_lr / start_lr, 1 / epochs))
# scheduler = lr_scheduler.StepLR(adam, 33, 0.1)
weight_ssim = 5e-2
weight_l2 = 1 - weight_ssim
rng = np.random.default_rng()

for epoch in trange(1, epochs + 1, ncols=80, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
    perm = np.arange(len(images)) # rng.permutation(len(images))
    for image, pose in tqdm(zip(images[perm], poses[perm]), ncols=80, total=len(images), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        image, pose = torch.from_numpy(image).to(device), torch.from_numpy(pose).to(device)
        ray_bundle = pose_to_ray_bundle_linear(pose, offset)
        ray_bundle.sample(0, 140 * 0.001, 80 * 0.001, image.shape[0], image.shape[1])
        rendered_image = render_ray_bundle(ray_bundle, nerf)
        
        loss_ssim = 1 - ms_ssim(rendered_image.unsqueeze(0).unsqueeze(0), image.unsqueeze(0).unsqueeze(0))
        loss_l2 = l2(rendered_image, image)
        loss = weight_ssim * loss_ssim + weight_l2 * loss_l2
        adam.zero_grad()
        loss.backward()
        adam.step()
    with torch.no_grad():
        side_by_side_image = np.concatenate([ (image.numpy(force=True) * 255).astype(np.uint8), (rendered_image.numpy(force=True) * 255).astype(np.uint8) ], axis=-1)
        # imageio.imwrite(f'./{epoch}.png', side_by_side_image)
        imageio.imwrite(f'.\logs\{epoch}.png', side_by_side_image)

    scheduler.step()

    if False:
        run.log({
            'ssim'  : loss_ssim.item(),
            'l2'    : loss_l2.item(),
            'loss'  : loss.item(),
            'lr'    : scheduler.get_last_lr()[0],
        })

    if epoch % 100 == 0 or epoch == epochs:
        torch.save(nerf.state_dict(), f'nerf_{epoch}.pt')
        torch.save(adam.state_dict(), f'adam_{epoch}.pt')

    

    
    





    