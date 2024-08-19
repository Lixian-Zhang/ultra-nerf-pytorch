import torch
import numpy as np
from tqdm import tqdm, trange
import imageio
import open3d as od
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
pcds = []
for image, pose in tqdm(zip(images, poses), total=len(poses)):
    print(pose)
    pcd = od.geometry.PointCloud()
    pose[:3, -1] *= 0.001
    image_pt, pose = torch.from_numpy(image).to(device), torch.from_numpy(pose).to(device)
    rb = pose_to_ray_bundle_linear(pose)
    rb.sample(0, 140 * 0.001, 80 * 0.001, image.shape[0], image.shape[1])
    pcd.points = od.utility.Vector3dVector(rb.points.numpy(force=True).reshape(-1, 3)[::100, :])
    c = (i % 255) / 255
    pcd.paint_uniform_color([c, c, c])
    pcds.append(pcd)
    i += 1
od.visualization.draw_geometries(pcds)
    

    
    





    