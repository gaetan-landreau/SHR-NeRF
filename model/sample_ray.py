import sys

sys.path.append("../")
import torch
import torch.nn.functional as F
import numpy as np
from math import sqrt, exp

from utils.nerf_helpers import get_rays_from_poses


class RaySampler(object):
    def __init__(self, data):
        super().__init__()
        self.instance_idx = data["index"]
        self.render_imgs = data["images"] if "images" in data.keys() else None
        self.render_poses = data["poses"]
        # self.render_masks = data['masks']
        self.render_bboxes = data["bboxes"]
        self.intrinsics = data["intrinsics"][:, 0]
        
        self.depth_range = data["depth_range"][0][0]

        self.src_view = data["src_view"]

        self.src_img = data["src_image"][:, 0]
        self.src_pose = data["src_pose"][:, 0]


        (
            self.num_objs,  # Batch size
            self.num_views_per_obj, # 50 in the training set. 
            _,
            self.H,
            self.W,
        ) = self.render_imgs.shape  # [B, N, 3, H, W]

        self.set_rays_and_rgb_gt()
    
    def set_rays_and_rgb_gt(self):
      
        all_rays_o = []
        all_rays_d = []
        for obj_idx in range(self.num_objs):
            #print(self.render_poses[obj_idx][10,:])
            rays_o, rays_d = get_rays_from_poses(
                self.H, self.W, self.intrinsics[obj_idx], self.render_poses[obj_idx]
            )
           
            rays_o = rays_o.reshape(-1, rays_o.shape[-1])  # [H*W*Nbviews,3]
            rays_d = rays_d.reshape(-1, rays_d.shape[-1])  # [H*W*Nbviews,3]

            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)

        all_rays_o = torch.stack(all_rays_o)  # [B, H*W*Nbviews, 3]
        all_rays_d = torch.stack(all_rays_d)  # [B, H*W*Nbviews, 3]
        all_rgb_gt = self.render_imgs.permute(0, 1, 3, 4, 2).reshape(
            self.num_objs, -1, 3
        )  # [B, H*W*Nbviews, 3]

        self.rays_o = all_rays_o
        self.rays_d = all_rays_d
        self.rgb_gt = all_rgb_gt

    def bbox_sample(self, obj_idx, obj_views_ids, N_rand):
        image_ids = torch.from_numpy(np.random.choice(obj_views_ids, (N_rand,)))
        pix_bboxes = self.render_bboxes[obj_idx][image_ids]
        x = (
            torch.rand(N_rand) * (pix_bboxes[:, 2] + 1 - pix_bboxes[:, 0])
            + pix_bboxes[:, 0]
        ).long()
        y = (
            torch.rand(N_rand) * (pix_bboxes[:, 3] + 1 - pix_bboxes[:, 1])
            + pix_bboxes[:, 1]
        ).long()
        pix = torch.stack((image_ids, y, x), dim=-1)
        return pix
    def symmetrize_pose(self,M):
        self.render_poses = torch.matmul(M,self.render_poses)
        self.set_rays_and_rgb_gt()
    # def get_all(self):
    #     ret = {'ray_o': self.rays_o.cuda(),
    #            'ray_d': self.rays_d.cuda(),
    #            'depth_range': self.depth_range.cuda(),
    #            'camera': self.camera.cuda(),
    #            'rgb': self.rgb.cuda() if self.rgb is not None else None,
    #            'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
    #            'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
    #     }
    #     return ret

    def get_all_single_image(self, index, device):
        select_inds = torch.arange(
            index * self.H * self.W, (index + 1) * self.H * self.W
        )

        rays_o = self.rays_o[:1, select_inds]
        rays_d = self.rays_d[:1, select_inds]
       
        rgb = self.render_imgs[:1, index]
        pose = self.render_poses[:1, index]

        ret = {
            "instance_idx": self.instance_idx.long().to(device),
            "rays_o": rays_o.to(device),
            "rays_d": rays_d.to(device),
            "pose": pose.to(device),
            "intrinsics": self.intrinsics[:1].to(device),
            "z_near": self.depth_range[0],
            "z_far": self.depth_range[1],
            "rgb": rgb.to(device),
            "image_size": torch.tensor([self.H, self.W], dtype=torch.float32).to(
                device
            ),
            "src_img": self.src_img[:1].to(device),
            "src_pose": self.src_pose[:1].to(device),
            # 'src_mask': self.src_mask[:1].cuda(),
            # 'src_bbox': self.src_bbox[:1].cuda()
        }
        return ret

    def get_all(self):
        pass

    def sample_random_pixel(self, obj_idx, N_rand, use_bbox):
        if not use_bbox:
            self.render_bboxes = None
        obj_views_ids = range(self.num_views_per_obj)
        
        # Go there at the begginig of the training. use_bbox = (global_step < args.bbox_step) in train.py 
        if self.render_bboxes is not None:
            pix = self.bbox_sample(obj_idx, obj_views_ids, N_rand)
            select_inds = (
                pix[..., 0] * self.H * self.W + pix[..., 1] * self.W + pix[..., 2]
            )
        # Finish the training there, without any constrains. 
        else:
            select_inds = torch.randint(
                0, self.num_views_per_obj * self.H * self.W, (N_rand,)
            )
            pix = None
        return pix, select_inds

    def random_sample(self, N_rand, device, use_bbox=False):
        rays_o = []
        rays_d = []
        rgb = []
        selected_indexes = []
        pixs = []
        #print(f'Value of self.num_objs: {self.num_objs}')
        #print(f'Shape of self.rgb_gt: {self.rgb_gt.shape}')
        #print(f'Value of num_views_per_obj: {self.num_views_per_obj}') 
        for obj_idx in range(self.num_objs):
            pix, select_inds = self.sample_random_pixel(obj_idx, N_rand, use_bbox)

            rays_o.append(self.rays_o[obj_idx][select_inds])
            rays_d.append(self.rays_d[obj_idx][select_inds])
            rgb.append(self.rgb_gt[obj_idx][select_inds])
            selected_indexes.append(select_inds)
            if pix is not None:
                pixs.append(pix)

        rays_o = torch.stack(rays_o)  # [B, N_rand, 3]
        rays_d = torch.stack(rays_d)  # [B, N_rand, 3]
        rgb = torch.stack(rgb)  # [B, N_rand, 3]

        ret = {
            "instance_idx": self.instance_idx,  # .long().cuda(),
            "rays_o": rays_o.to(device),
            "rays_d": rays_d.to(device),
            "intrinsics": self.intrinsics.to(device),
            "z_near": self.depth_range[0],
            "z_far": self.depth_range[1],
            "rgb": rgb.to(device),
            "image_size": torch.tensor([self.H, self.W], dtype=torch.float32).to(
                device
            ),
            "src_img": self.src_img.to(device),
            "src_pose": self.src_pose.to(device),
            "select_inds": selected_indexes,
            "pixs": pixs
            # 'src_mask': self.src_mask.cuda(),
            # 'src_bbox': self.src_bbox.cuda(),
        }
        return ret


