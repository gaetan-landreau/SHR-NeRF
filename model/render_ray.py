import sys
sys.path.append('../')
import torch
import torch.nn.functional as F

from utils.nerf_helpers import *
from network.nerf import run_nerf_symm_local
from utils.general import *
from einops import rearrange, repeat



def sample_along_camera_ray(rays_o, rays_d, z_near, z_far, device,
                            N_samples, lindisp=False, det=False):
    num_objs, N_rays = rays_o.shape[:2]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = z_near * (1. - t_vals) + z_far * t_vals
    else:
        z_vals = 1. / (1. / z_near * (1. - t_vals) + 1. / z_far * t_vals)

    z_vals = z_vals.expand([num_objs, N_rays, N_samples])
    
    if not det:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    z_vals = z_vals.to(device)

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    return pts, viewdirs, z_vals

def get_symmetric_points_and_directions(pts,viewdirs,M):

    xyz = torch.reshape(pts, [pts.shape[0], -1, pts.shape[-1]])
    xyz_symm = (M @ xyz.unsqueeze(-1))[..., 0]
    pts_s = torch.reshape(xyz_symm, pts.shape)

    viewdirs_symm = (M @ viewdirs.unsqueeze(-1))[..., 0]

    return pts_s, viewdirs_symm

# Hierarchical sampling
def sample_pdf(bins, weights, N_samples, det=False):
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous().to(device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], inds_g.shape[2], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(2).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(2).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def run_network(inputs, viewdirs, networks, embed_fn, embeddirs_fn, local_feature,cosine_mod,ret=None):
    inputs_flat = torch.reshape(inputs, [inputs.shape[0], -1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    input_dirs = viewdirs[:, :, None].expand(inputs.shape)

    input_dirs_flat = torch.reshape(input_dirs, [input_dirs.shape[0], -1, input_dirs.shape[-1]])
    embedded_dirs = embeddirs_fn(input_dirs_flat)
    embedded = torch.cat([embedded, embedded_dirs], -1)

    if local_feature is not None:
        embedded = torch.cat([embedded, local_feature], -1)
        
    if cosine_mod['use']:
        #print(f'Shape of embedded: {embedded.shape}')
        #print(f'Shape of module_with_cosine: {module_with_cosine["cs"].shape}')
        #print(f'Shape of local_feature: {local_feature.shape}')
        embedded = torch.cat([embedded, cosine_mod['cs']], -1)
        
    nerf = networks['nerf']
    outputs_flat = nerf(embedded)
    
    ray_transformer = networks['ray_transformer'] 
    
    if ray_transformer is not None:
        alpha_raw = outputs_flat[..., 3:]
        
        # From MatchNeRF 
        mask = (ret['uv'] > -1.0)*(ret['uv']<1.0)
        mask = (mask[...,0]*mask[...,1]).float()
        
        num_valid_obs = torch.sum(mask, dim=-1, keepdim=True)
        
        _,_,alpha = ray_transformer(alpha_raw,alpha_raw,alpha_raw,mask=(num_valid_obs>1).float())
        #print(f'Shape of alpha: {alpha.shape}')
        outputs_flat = torch.cat([outputs_flat[..., :3], alpha], -1)

    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    #print(f'Re-shaped output shape: {outputs.shape}')
    return outputs


def run_nerf_through_hypernetwork(pts, viewdirs, model,ret_features,latent_vector,which_nerf):
        
        # Networks. 
        networks = {'nerf':None,
                    'ray_transformer':model.ray_transformer if hasattr(model,'ray_transformer') else None}

      
        feature_type = model.feature_type
        
        local_feature = build_local_feature(ret_features,feature_type)
        
        _feat_ch = local_feature.shape[-1]

        assert which_nerf in ['coarse','fine']
        nerf_layers = model.hypernetwork(latent_vector) if which_nerf =='coarse' else model.hypernetwork_fine(latent_vector)
        
        nerf = lambda x: run_nerf_symm_local(x, nerf_layers=nerf_layers, input_ch=model.input_ch,
                                                    input_ch_views=model.input_ch_views,
                                                    local_feature_ch=_feat_ch,
                                                    cosine_mod = {'use':False})
        networks['nerf'] = nerf

        raw = run_network(pts, viewdirs, networks,
                                model.embed_fn, model.embeddirs_fn,
                                local_feature,{'use':False},ret_features)
        
        return raw
    
def run_nerf_through_resnet(pts,ray_batch, model,ret_features,which_nerf):
        
        N_rays = pts.shape[1]
        N_samples = pts.shape[2]
        batch = pts.shape[0]
        
        f = ret_features['feat']
        f_s= ret_features['feat_s']
        rgb = ret_features['rgb']
        
        xyz_c = ret_features['xyz_c'] # 3D points sampled from target view expressed in source view coordinates.
        xyz_c = rearrange(xyz_c,'b (nr ns) ch -> b nr ns ch', nr=N_rays,ns= N_samples)
        
        dirs = repeat(ray_batch['rays_d'], 'b nr c -> b nr ns c', ns=N_samples)
        dir_c = model.feature_net.compute_dir(dirs, ray_batch)  
        
        rgb= rearrange(rgb,'b (nr ns) rgb -> b nr ns rgb', nr=N_rays,ns= N_samples)
        feat = rearrange(f,'b (nr ns) c -> b nr ns c', nr=N_rays,ns= N_samples)
        feat_s = rearrange(f_s,'b (nr ns) c -> b nr ns c', nr=N_rays,ns= N_samples)
     
        feat_in = torch.cat([feat,feat_s,rgb,dir_c],-1)
        
        assert which_nerf in ['coarse','fine']
        raw = model.coarse_net(xyz_c.flatten(0, 1), feat_in.flatten(0, 1)) if which_nerf =='coarse' else \
              model.fine_net(xyz_c.flatten(0,1),feat_in.flatten(0,1))
        raw = raw.reshape([batch, N_rays, N_samples, 4])
        
        return raw
    
def render_rays(ray_batch, model, device, latent_vector,enforceSymm, N_samples,cosine_mod,use_ray_transformer, lindisp=False, N_importance=0,
                det=False, raw_noise_std=0., white_bkgd=False, shape_codes=None, texture_codes=None, noise=None):
    ret = {'outputs_coarse': None,
           'outputs_fine': None,
           'outputs_coarse_s':None,
           'outputs_fine_s':None}
    
    rays_o = ray_batch['rays_o']
    rays_d = ray_batch['rays_d']
    z_near = ray_batch['z_near']
    z_far = ray_batch['z_far']
    
    pts, viewdirs, z_vals = sample_along_camera_ray(rays_o=rays_o,
                                                    rays_d=rays_d,
                                                    z_near=z_near,
                                                    z_far=z_far,
                                                    device=device,
                                                    N_samples=N_samples, lindisp=lindisp, det=det)
    
    use_hypernetworks = hasattr(model, 'hypernetwork')
    
    ret_coarse = model.feature_net.index(pts,ray_batch)
    
    symmetrize_sampling = (enforceSymm['status'] and model.mode == 'train' and np.random.rand() < 0.40)
    
    if symmetrize_sampling:
        M = model.get_symmetry_matrix()
        ptsS, viewdirsS = get_symmetric_points_and_directions(pts,viewdirs,M)
        
    raw_coarse = run_nerf_through_hypernetwork(pts, viewdirs, model,ret_coarse,latent_vector,which_nerf ='coarse') if (use_hypernetworks and not symmetrize_sampling) else \
                 run_nerf_through_hypernetwork(ptsS, viewdirsS, model,ret_coarse,latent_vector,which_nerf ='coarse') if (use_hypernetworks and symmetrize_sampling) else \
                 run_nerf_through_resnet(pts, ray_batch, model,ret_coarse,which_nerf ='coarse')
                
    
    outputs_coarse = raw2outputs(raw_coarse, z_vals, ray_batch['rays_d'],
                                 device=device,
                                 raw_noise_std=raw_noise_std,
                                 white_bkgd=white_bkgd,
                                 use_hypernetworks=use_hypernetworks)

    ret['outputs_coarse'] = outputs_coarse

    if N_importance > 0:
        # detach since we would like to decouple the coarse and fine networks
        weights = outputs_coarse['weights'].clone().detach()            # [N_rays, N_samples]
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=det)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
        
        ret_fine = model.feature_net.index(pts, ray_batch)
        
        if symmetrize_sampling:
            M = model.get_symmetry_matrix()
            ptsS, viewdirsS = get_symmetric_points_and_directions(pts,viewdirs,M)
           
        raw_fine = run_nerf_through_hypernetwork(pts, viewdirs, model,ret_fine,latent_vector,which_nerf ='fine') if (use_hypernetworks and not symmetrize_sampling) else \
                   run_nerf_through_hypernetwork(ptsS, viewdirsS, model,ret_fine,latent_vector,which_nerf ='fine') if (use_hypernetworks and symmetrize_sampling) else \
                   run_nerf_through_resnet(pts, ray_batch, model,ret_fine,which_nerf ='fine')
      
        #####################################################################################
        outputs_fine = raw2outputs(raw_fine, z_vals, ray_batch['rays_d'],
                                   device=device,
                                   raw_noise_std=raw_noise_std,
                                   white_bkgd=white_bkgd,
                                   use_hypernetworks=use_hypernetworks)
        ret['outputs_fine'] = outputs_fine

    return ret


def raw2outputs(raw, z_vals, rays_d, device, raw_noise_std=0., white_bkgd=False,use_hypernetworks=True):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(device)], -1)  # [batch,N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  if use_hypernetworks else raw[...,:3] # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[..., :-1]
    weights = alpha * T 
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    ret = {
        'rgb': rgb_map,
        'disp': disp_map,
        'acc': acc_map,
        'weights': weights,
        'depth': depth_map,
        'alpha': alpha,
        'T':T,
        'sigma': F.relu(raw[..., 3]),
    }

    return ret


def render_single_image(ray_sampler, ray_batch, model, device, latent_vector,enforceSymm,cosine_mod, use_ray_transformer,chunk_size, N_samples,
                        lindisp=False, N_importance=0, det=False, white_bkgd=False, shape_codes=None, texture_codes=None,
                        noise=None):
    all_ret = {'outputs_coarse': {},
               'outputs_fine': {},
               'outputs_coarse_s':{},
               'outputs_fine_s':{}}

    N_rays = ray_batch['rays_o'][0].shape[0]

    for i in range(0, N_rays, chunk_size):
        chunk = {}
        for k in ray_batch:
            if k in ['pose', 'z_near', 'z_far', 'src_img', 'src_pose',
                     'image_size', 'intrinsics', 'rgb', 'instance_idx']:
                chunk[k] = ray_batch[k]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][:, i:i+chunk_size]
            else:
                chunk[k] = None

        ret = render_rays(ray_batch=chunk,
                          model=model,
                          device=device,
                          latent_vector=latent_vector,
                          enforceSymm = enforceSymm,
                          N_samples=N_samples,
                          cosine_mod=cosine_mod,
                          use_ray_transformer=use_ray_transformer,
                          lindisp=lindisp,
                          N_importance=N_importance,
                          det=det,
                          white_bkgd=white_bkgd,
                          shape_codes=shape_codes,
                          texture_codes=texture_codes,
                          noise=noise)

        # Handle both coarse and fine outputs
        # Cache chunk results on cpu
        # Handle both coarse and fine outputs
        # Cache chunk results on cpu
        if i == 0:
            for k in ret['outputs_coarse']:
                all_ret['outputs_coarse'][k] = []

            if ret['outputs_fine'] is None:
                all_ret['outputs_fine'] = None
            else:
                for k in ret['outputs_fine']:
                    all_ret['outputs_fine'][k] = []

            if ret['outputs_coarse_s'] is None:
                all_ret['outputs_coarse_s'] = None
            else:
                for k in ret['outputs_coarse_s']:
                    all_ret['outputs_coarse_s'][k] = []

            if ret['outputs_fine_s'] is None:
                all_ret['outputs_fine_s'] = None
            else:
                for k in ret['outputs_fine_s']:
                    all_ret['outputs_fine_s'][k] = []


        for k in ret['outputs_coarse']:
            all_ret['outputs_coarse'][k].append(ret['outputs_coarse'][k].cpu())

        if ret['outputs_fine'] is not None:
            for k in ret['outputs_fine']:
                all_ret['outputs_fine'][k].append(ret['outputs_fine'][k].cpu())

        if ret['outputs_coarse_s'] is not None:
            for k in ret['outputs_coarse_s']:
                all_ret['outputs_coarse_s'][k].append(ret['outputs_coarse_s'][k].cpu())

        if ret['outputs_fine_s'] is not None:
            for k in ret['outputs_fine_s']:
                all_ret['outputs_fine_s'][k].append(ret['outputs_fine_s'][k].cpu())

    # Merge chunk results and reshape
    for k in all_ret['outputs_coarse']:
        tmp = torch.cat(all_ret['outputs_coarse'][k], dim=1).reshape(ray_sampler.H, ray_sampler.W, -1)
        all_ret['outputs_coarse'][k] = tmp.squeeze()

    if all_ret['outputs_fine'] is not None:
        for k in all_ret['outputs_fine']:
            tmp = torch.cat(all_ret['outputs_fine'][k], dim=1).reshape(ray_sampler.H, ray_sampler.W, -1)
            all_ret['outputs_fine'][k] = tmp.squeeze()

    if all_ret['outputs_coarse_s'] is not None:
        for k in all_ret['outputs_coarse_s']:
            tmp = torch.cat(all_ret['outputs_coarse_s'][k], dim=1).reshape(ray_sampler.H, ray_sampler.W, -1)
            all_ret['outputs_coarse_s'][k] = tmp.squeeze()

    if all_ret['outputs_fine_s'] is not None:
        for k in all_ret['outputs_fine_s']:
            tmp = torch.cat(all_ret['outputs_fine_s'][k], dim=1).reshape(ray_sampler.H, ray_sampler.W, -1)
            all_ret['outputs_fine_s'][k] = tmp.squeeze()

    return all_ret


def log_view_to_tb(writer, global_step, args, model, device, ray_sampler, render_view, gt_img, prefix=''):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all_single_image(render_view,device)

        latent_vector = model.encode(ray_batch['src_img'])

        ret = render_single_image(ray_sampler=ray_sampler,
                                  ray_batch=ray_batch,
                                  model=model,
                                  device=device,
                                  latent_vector=latent_vector,
                                  enforceSymm = args.enforce_symmetry,
                                  cosine_mod = args.cosine_mod,
                                  use_ray_transformer=args.use_ray_transformer,
                                  chunk_size=args.chunk_size,
                                  N_samples=args.N_samples,
                                  lindisp=args.lindisp,
                                  det=True,
                                  N_importance=args.N_importance,
                                  white_bkgd=args.white_bkgd)
    rgb_src = ray_batch['src_img'][0].detach().cpu()
    rgb_gt = img_HWC2CHW(gt_img)
    rgb_pred = img_HWC2CHW(ret['outputs_coarse']['rgb'].detach().cpu())

    rgb_im = torch.dstack([rgb_src, rgb_gt, rgb_pred])
    depth_im = ret['outputs_coarse']['depth'].detach().cpu()
    acc_map = ret['outputs_coarse']['acc'].detach().cpu()

    if ret['outputs_fine'] is None:
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=False))
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
    else:
        rgb_fine = img_HWC2CHW(ret['outputs_fine']['rgb'].detach().cpu())
        rgb_im = torch.cat((rgb_im, rgb_fine), dim=-1)
        depth_im = torch.cat((depth_im, ret['outputs_fine']['depth'].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=False))
        acc_map = torch.cat((acc_map, torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()), dim=-1)
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))

    # Write the pred/gt rgb images and depths
    writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im, global_step)
    writer.add_image(prefix + 'depth-coarse-fine', depth_im, global_step)
    writer.add_image(prefix + 'acc-coarse-fine', acc_map, global_step)

    # Write scalar
    pred_rgb = ret['outputs_fine']['rgb'] if ret['outputs_fine'] is not None else ret['outputs_coarse']['rgb']
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    model.switch_to_train()
    return rgb_im, depth_im, acc_map
