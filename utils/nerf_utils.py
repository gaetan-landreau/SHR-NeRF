import torch 
import torch.nn.functional as F


def query_cond_info(self, point_samples, ref_poses, ref_images, ref_feats_list):
	'''
			query conditional information from reference images, using the target position.
			point_samples: B, n_rays, n_samples, 3
			ref_poses: dict, all camera information of reference images 
						'extrinsics': B, N, 3, 4; 'intrinsics': B, N, 3, 3; 'near_fars': B, N, 2
			ref_images: B, n_views, 3, H, W. range: [0, 1] !!!
	'''
		
	# What reference images are ? The source one ? 
	# Focus here on mask_info. 
	batch_size, n_views, _, img_h, img_w = ref_images.shape
	assert ref_feats_list is not None, "Must provide the image feature for info query."

	device = self.opts.device
	cos_n_group = self.opts.encoder.cos_n_group
	cos_n_group = [cos_n_group] if isinstance(cos_n_group, int) else cos_n_group
	feat_data_list = [[] for _ in range(len(ref_feats_list))]
	color_data = []
	mask_data = []

	# query information from each source view
	inv_scale = torch.tensor([[img_w - 1, img_h - 1]]).repeat(batch_size, 1).to(device)
	for view_idx in range(n_views):
		near_far_ref = ref_poses['near_fars'][:, view_idx]
		extr_ref, intr_ref = ref_poses['extrinsics'][:, view_idx].clone(), ref_poses['intrinsics'][:, view_idx].clone()
		point_samples_pixel = get_coord_ref_ndc(extr_ref, intr_ref, point_samples,
														   inv_scale, near_far=near_far_ref)
		grid = point_samples_pixel[..., :2] * 2.0 - 1.0

		# query enhanced features infomation from each view
		for scale_idx, img_feat_cur_scale in enumerate(ref_feats_list):
			raw_whole_feats = img_feat_cur_scale[:, view_idx]
			sampled_feats = sample_features_by_grid(raw_whole_feats, grid, align_corners=True, mode='bilinear', padding_mode='border',
														local_radius=self.opts.encoder.feature_sample_local_radius,
														local_dilation=self.opts.encoder.feature_sample_local_dilation)
			feat_data_list[scale_idx].append(sampled_feats)

		# query color
		color_data.append(F.grid_sample(ref_images[:, view_idx], grid, align_corners=True, mode='bilinear', padding_mode='border'))
		
		# record visibility mask for further usage
		in_mask = ((grid > -1.0) * (grid < 1.0))
		in_mask = (in_mask[..., 0] * in_mask[..., 1]).float()
		mask_data.append(in_mask.unsqueeze(1))
	all_data = {}
	
	# merge queried information from all views
	
	# merge extracted enhanced features
	merged_feat_data = []
	for feat_data_idx, raw_feat_data in enumerate(feat_data_list):  # loop over scale
		cur_updated_feat_data = []
		# split back to original
		split_feat_data = [torch.split(x, int(x.shape[1] / (n_views - 1)), dim=1) for x in raw_feat_data]
		# calculate simliarity for feature from the same transformer
		index_lists = [(a, b) for a in range(n_views - 1) for b in range(a, n_views - 1)]
		for i_idx, j_idx in index_lists:
			input_a = split_feat_data[i_idx][j_idx]  # B x C x N_rays x N_pts
			input_b = split_feat_data[j_idx + 1][i_idx]
			iB, iC, iR, iP = input_a.shape
			group_a = input_a.reshape(iB, cos_n_group[feat_data_idx], int(iC / cos_n_group[feat_data_idx]), iR, iP)
			group_b = input_b.reshape(iB, cos_n_group[feat_data_idx], int(iC / cos_n_group[feat_data_idx]), iR, iP)
			cur_updated_feat_data.append(torch.nn.CosineSimilarity(dim=2)(group_a, group_b))
		cur_updated_feat_data = torch.stack(cur_updated_feat_data, dim=1)  # [B, n_pairs, n_groups, n_rays, n_pts]

		cur_updated_feat_data = torch.mean(cur_updated_feat_data, dim=1, keepdim=True)
		cur_updated_feat_data = cur_updated_feat_data.reshape(cur_updated_feat_data.shape[0], -1, *cur_updated_feat_data.shape[-2:])
		merged_feat_data.append(cur_updated_feat_data)

	merged_feat_data = torch.cat(merged_feat_data, dim=1)
	# all_data.append(merged_feat_data)
	all_data['feat_info'] = merged_feat_data

	# merge extracted color data
	merged_color_data = torch.cat(color_data, dim=1)
	# all_data.append(merged_color_data)
	all_data['color_info'] = merged_color_data

	# merge visibility masks
	merged_mask_data = torch.cat(mask_data, dim=1)
	# all_data.append(merged_mask_data)
	all_data['mask_info'] = merged_mask_data

	# all_data = torch.cat(all_data, dim=1)[0].permute(1, 2, 0)
	for k, v in all_data.items():
		all_data[k] = v.permute(0, 2, 3, 1)  # (b, n_rays, n_samples, n_dim)

	return all_data





def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom

# basic operations of transforming 3D points between world/camera/image coordinates


def world2cam(X, pose):  # [B,N,3]
    X_hom = to_hom(X)
    return X_hom@pose.transpose(-1, -2)


def get_coord_ref_ndc(extr_ref, intr_ref, pts_3D, inv_scale, near_far=None, lindisp=False):
    '''
        Warp the provided position to the reference coordinate, and normalize to NDC coordinate.
        pts_3D [batch, N_rays N_sample 3]
    '''

    bs, N_rays, N_samples, N_dim = pts_3D.shape
    pts_3D = pts_3D.reshape(bs, -1, N_dim)
    near, far = torch.split(near_far, [1, 1], dim=-1)

    # wrap to ref view
    if extr_ref is not None:
        pts_ref_world = world2cam(pts_3D, extr_ref)

    if intr_ref is not None:
        # using projection
        point_samples_pixel = pts_ref_world @ intr_ref.transpose(-1, -2)
        point_samples_pixel[..., :2] = (point_samples_pixel[..., :2] / point_samples_pixel[..., -1:] + 0.0) / inv_scale.reshape(bs, 1, 2)  # normalize to 0~1
        if not lindisp:
            point_samples_pixel[..., 2] = (point_samples_pixel[..., 2] - near) / (far - near)  # normalize to 0~1
        else:
            point_samples_pixel[..., 2] = (1.0/point_samples_pixel[..., 2]-1.0/near)/(1.0/far - 1.0/near)
    else:
        # using bounding box
        near, far = near.view(bs, 1, 3), far.view(bs, 1, 3)
        point_samples_pixel = (pts_ref_world - near) / (far - near)  # normalize to 0~1

    point_samples_pixel = point_samples_pixel.view(bs, N_rays, N_samples, 3)
    return point_samples_pixel