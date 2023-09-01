import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.deeplabv3.decoder import ASPP, SeparableConv2d
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange, repeat

def remove_bn_dropout(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        #print("removed batch norm")
        module_output = torch.nn.Identity()
    elif isinstance(module, torch.nn.modules.Conv2d):
        module_output = torch.nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups, bias=True, padding_mode=module.padding_mode)
        with torch.no_grad():
            module_output.weight = module.weight
            if module.bias == None:
                module_output.bias = torch.nn.Parameter( torch.zeros_like(module_output.bias) )
            else:
                module_output.bias = module.bias
    elif isinstance(module, torch.nn.modules.Dropout):
        #print("removed dropout")
        module_output = torch.nn.Identity()

    for name, child in module.named_children():
        module_output.add_module(
            name, remove_bn_dropout(child)
        )
    del module
    return module_output

class DeepLabV3PlusDecoderHR(torch.nn.Module):
    def __init__(
        self,
        encoder_channels,
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16,
        encoder_in_channels=3,
    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))
        
        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = torch.nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = torch.nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(highres_out_channels),
            torch.nn.ReLU(),
        )
        self.block2 = torch.nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )


        in_channels = 256
        # out_channels = 1024

        self.block3 = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2.0),
            torch.nn.Conv2d(in_channels, in_channels // 2, 3,padding=1),
            torch.nn.ReLU(),
        ) # 128 x 64 x 64
        self.projblock3 = torch.nn.Conv2d(64, 128, 1)

        self.block4 = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2.0),
            torch.nn.Conv2d(in_channels // 2, in_channels // 2, 3,padding=1),
            torch.nn.ReLU(),
        ) # 128 x 128 x 128
        self.projblock4 = torch.nn.Conv2d(encoder_in_channels, 128, 1)


    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features_tmp = self.block2(concat_features)
        
        fused_features = self.block3(fused_features_tmp) + self.projblock3(features[-5])
        
        fused_features_tmp = self.block4(fused_features) + self.projblock4(features[-6])
       
        return fused_features_tmp


def get_encoder_deeplab(in_channels=3, out_channels=16*16, replace_decoder=False, encoder_name='resnet34'):
    model = smp.DeepLabV3Plus(in_channels=in_channels, classes=out_channels, encoder_name=encoder_name)
    model.decoder = DeepLabV3PlusDecoderHR(
            encoder_channels=model.encoder.out_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
            encoder_in_channels=in_channels,
        )

    model.segmentation_head = torch.nn.Sequential(
        torch.nn.Conv2d(256 // 2, out_channels, 3, padding=1)
    )

    return remove_bn_dropout(model)

class deepLabv3Plus(nn.Module):
    def __init__(self,in_channels=3, out_channels=16*16, encoder_name='resnet34',**kwargs):
        
        super(deepLabv3Plus, self).__init__()
        
        self.model = smp.DeepLabV3Plus(in_channels=in_channels, classes=out_channels, encoder_name=encoder_name,encoder_weights='imagenet')
        self.model.decoder = DeepLabV3PlusDecoderHR(encoder_channels=self.model.encoder.out_channels,
                                               out_channels=256,
                                               atrous_rates=(12, 24, 36),
                                               output_stride=16,
                                               encoder_in_channels=in_channels,
                                               )
        self.model.segmentation_head = torch.nn.Sequential(torch.nn.Conv2d(256 // 2, out_channels, 3, padding=1))
        
        self.model = remove_bn_dropout(self.model)
        
  
        self.add_high_res_skip = kwargs.get('add_high_res_skip',False)
        
        # Add an additional CNN for high res. skip connection. 
        if self.add_high_res_skip: 
            self.conv_map = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,out_channels)
    
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer("latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False)
        
        self.index_interp="bilinear"
        self.index_padding="zeros"
        
        
    def forward(self,x):
        
        latents = self.model.encoder(x)
        latent_inter = self.model.decoder(*latents)
       
        self.latent = self.model.segmentation_head(latent_inter)
        if self.add_high_res_skip:
            hr_skip = self.conv_map(x)
            self.latent = torch.concat([self.latent, hr_skip], dim = 1)
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0

        z = self.avgpool(latents[-1])
        z = torch.flatten(z, 1)
        z = self.fc(z)

            
        return z
    
    
    def compute_dir(self,dirs, ray_batch):
        
        train_exts = ray_batch['src_pose']
        _, N_rays, N_samples, _ = dirs.shape
   
        dirs = rearrange(dirs, 'b nr ns c -> b c (nr ns)')
        train_poses = train_exts[..., :3, :3]  # [batch, n_views, 4, 4]
    
        dir_c = torch.inverse(train_poses) @ (dirs)
        dir_c = rearrange(dir_c, 'b c (nr ns) -> b nr ns c', nr=N_rays, ns=N_samples)
        return dir_c
    
    def index(self, xyz, ray_batch):
        xyz = torch.reshape(xyz, [xyz.shape[0], -1, xyz.shape[-1]])
        # Models in ShapeNet have already been processed so that in their canonical poses,
        # the Y-Z plane is the plane of the reflection symmetry

        ret = {
            "uv": None,
            "uv_symm": None,
            "feat": None,
            "feat_s": None,
        }

        M = torch.tensor([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32).to(
            xyz.device
        )
          
        src_poses = ray_batch['src_pose']
        intrinsics = ray_batch['intrinsics']
        image_size = ray_batch['image_size']
        src_image = ray_batch['src_img'] 
      
        xyz_symm = (M @ xyz.unsqueeze(-1))[..., 0]

        focal = intrinsics[:, 0, 0].view(-1, 1, 1).repeat(1, xyz.shape[-2], 2)
        focal[..., 1] = focal[..., 1] * -1
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        c = torch.stack([cx, cy], -1).unsqueeze(1).repeat(1, xyz.shape[-2], 1)
       
        rot = src_poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, src_poses[:, :3, 3:])  # (B, 3, 1)
        w2c = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)
        # Transform query points into the camera space of the source view
        xyz_rot = torch.matmul(w2c[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
        xyz = xyz_rot + w2c[:, None, :3, 3]
        xyz_symm_rot = torch.matmul(w2c[:, None, :3, :3], xyz_symm.unsqueeze(-1))[
            ..., 0
        ]
        xyz_symm = xyz_symm_rot + w2c[:, None, :3, 3]

        uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
        uv = uv * focal + c

        uv_symm = -xyz_symm[:, :, :2] / xyz_symm[:, :, 2:]  # (SB, B, 2)
        uv_symm = uv_symm * focal + c

        scale = self.latent_scaling / image_size
       
        uv = uv * scale - 1.0
        uv_symm = uv_symm * scale - 1.0

        uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        uv_symm = uv_symm.unsqueeze(2)  # (B, N, 1, 2)

        ret["uv"] = uv
        ret["uv_symm"] = uv_symm

        ret['xyz_c'] = xyz
        ret['xyz_c_symm'] = xyz_symm
        
        samples = F.grid_sample(self.latent,uv,align_corners=True,mode=self.index_interp,padding_mode=self.index_padding,)
        samples_symm = F.grid_sample(self.latent,uv_symm,align_corners=True,mode=self.index_interp,padding_mode=self.index_padding,)
        samples_rgb = F.grid_sample(src_image,uv,align_corners=True,mode=self.index_interp,padding_mode=self.index_padding,)
        
        feat = samples[:, :, :, 0].transpose(1, 2)
        feat_s = samples_symm[:, :, :, 0].transpose(1, 2)
        rgb = samples_rgb[:,:,:,0].transpose(1,2)

        ret["feat"] = feat
        ret["feat_s"] = feat_s
        ret['rgb'] = rgb
        
        return ret
    
        
    
if __name__ =='__main__':
    
    
    model = deepLabv3Plus()
    
    x = torch.zeros(1,3,187, 281)
    x = torch.zeros(1,3,192, 256)
    z = model(x)
    print(z.shape)