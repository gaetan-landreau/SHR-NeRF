import sys

sys.path.append("../")
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
import imageio
import numpy as np
from utils.data_utils import get_image_to_tensor, get_mask_to_tensor
from utils.general import parse_comma_separated_integers, pick


class SRNsDatasetOverfit(Dataset):
    def __init__(
        self,
        args,
        mode,
        nb_iter,
        scene="cars",
        image_size=(128, 128),
        world_scale=1.0,
    ):  # For few-shot case: Can pick specific observations only
        super().__init__()
        self.args = args
       
        self.nb_iter = nb_iter
        self.folder_path = os.path.join(args.datadir, scene + "_{}".format(mode))
        
        print("[Info] Loading SRNs dataset for overfitting: {}".format(self.folder_path))
        self.mode = mode
        self.scene = scene
        
        self.all_ids = sorted(glob.glob(os.path.join(self.folder_path, "*")))
        
        self.specific_observation_idx = np.random.randint(len(self.all_ids))
        self.specific_observation_id = self.all_ids[self.specific_observation_idx].split("/")[-1]
        
        self.intrinsics_file = os.path.join(self.folder_path,self.specific_observation_id, "intrinsics.txt")
        self.all_obj = sorted(glob.glob(os.path.join(self.folder_path,self.specific_observation_id, "*")))
        
        
        self.image_to_tensor = get_image_to_tensor()
        self.mask_to_tensor = get_mask_to_tensor()
        self.image_size = image_size
        self.world_scale = world_scale
        
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        self.z_near = 0.8
        self.z_far = 1.8

        self.initiate_data()
        
    def initiate_data(self):
        
        self.obj_dir = os.path.dirname(self.intrinsics_file)
        self.rgb_paths = sorted(glob.glob(os.path.join(self.obj_dir, "rgb", "*")))
        self.pose_paths = sorted(glob.glob(os.path.join(self.obj_dir, "pose", "*")))
        
        assert len(self.rgb_paths) == len(self.pose_paths)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        for rgb_path, pose_path in zip(self.rgb_paths, self.pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )
            pose = pose @ self._coord_trans

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            # if len(rnz) == 0:
            #     raise RuntimeError("ERROR: Bad image at", rgb_path, "please investigate!")
            try:
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
            except:
                bbox = torch.tensor([0, 0, 0, 0], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        self.all_imgs = torch.stack(all_imgs)
        self.all_poses = torch.stack(all_poses)
        self.all_masks = torch.stack(all_masks)
        self.all_bboxes = torch.stack(all_bboxes)

        with open(self.intrinsics_file, "r") as intrinsics:
            lines = intrinsics.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            H, W = map(int, lines[-1].split())
            
        # Intrinsics.
        K = np.array([[focal, 0.0, cx, 0.0], [0.0, focal, cy, 0], [0.0, 0, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
        
        self.intrinsics = torch.from_numpy(K).view(1, 4, 4).repeat(len(self.all_imgs), 1, 1)
        self.depth_range = (
            torch.tensor([self.z_near, self.z_far], dtype=torch.float32)
            .view(1, -1)
            .repeat(len(self.all_imgs), 1)
        )
        
        if (H, W) != self.image_size:
            scale = self.image_size[0] / H
            focal *= scale
            cx *= scale
            cy *= scale
            self.all_bboxes *= scale

            self.all_imgs = F.interpolate(self.all_imgs, size=self.image_size, mode="area")
            self.all_masks = F.interpolate(self.all_masks, size=self.image_size, mode="area")
        
            
    def __len__(self):
        return self.nb_iter #len(self.all_intrinsics_files)

    def __getitem__(self, index):
        # Select source view.
        src_view = np.random.choice(len(self.rgb_paths), 1)[0]
        
        # Get the corresponding image, pose and path. 
        src_img = self.all_imgs[src_view : src_view + 1]
        src_pose = self.all_poses[src_view : src_view + 1]
        src_path = self.rgb_paths[src_view]

        ret = {
            "obj_dir": self.obj_dir,
            "index": index,
            "intrinsics": self.intrinsics,
            "rgb_paths": self.rgb_paths,
            "images": self.all_imgs,
            "masks": self.all_masks,
            "bboxes": self.all_bboxes,
            "poses": self.all_poses,
            "depth_range": self.depth_range,
            "src_view": src_view,
            "src_path": src_path,
            "src_image": src_img,

            "src_pose": src_pose,
           
        }

        return ret
