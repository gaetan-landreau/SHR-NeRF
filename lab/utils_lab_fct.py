import cv2
import torch
import math
import numpy as np
from model.sample_ray import RaySampler
from skimage import metrics
from matplotlib import collections
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt


def renormalize_feature(uv, H, W):
    u = uv[:, 0]
    v = uv[:, 1]

    x = ((u + 1) * W) / 2.0
    y = ((v + 1) * H) / 2.0

    x = x.to(torch.uint8)
    y = y.to(torch.uint8)
    return x.squeeze(), y.squeeze()


def np_from_tensor(img_tensor):
    return img_tensor.cpu().detach().numpy()


def draw_projected_ray_on_F0(x, y, x_s, y_s, F0, nb_sampled_points_on_rays):
    count = 0
    count_s = 0
    F0_cp = F0.copy()
    for i in range(nb_sampled_points_on_rays):
        u = x[i]
        v = y[i]

        u_s = x_s[i]
        v_s = y_s[i]

        if u < 64 and v < 64:
            count += 1
            F0_cp[v, u] = [255, 0, 0]
        if u_s < 64 and v_s < 64:
            F0_cp[v_s, u_s] = [0, 255, 0]
            count_s += 1
    print(f"Number of valid sample: {count}")
    print(f"Number of symmetric valid sample: {count_s}")
    return F0_cp


def get_min_max_pix_loc(min_ray_index, max_ray_index, x, y, x_s, y_s):
    keys_coord = [
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "xmin_s",
        "xmax_s",
        "ymin_s",
        "ymax_s",
    ]
    res_coord = dict.fromkeys(keys_coord)

    # Get the corresponding (xy) coordinates that has minimize f_uv - f_uv_sym
    x_min, y_min = x[min_ray_index].cpu().numpy(), y[min_ray_index].cpu().numpy()
    x_s_min, y_s_min = (
        x_s[min_ray_index].cpu().numpy(),
        y_s[min_ray_index].cpu().numpy(),
    )

    x_max, y_max = x[max_ray_index].cpu().numpy(), y[max_ray_index].cpu().numpy()
    x_s_max, y_s_max = (
        x_s[max_ray_index].cpu().numpy(),
        y_s[max_ray_index].cpu().numpy(),
    )

    res_coord["xmin"] = x_min
    res_coord["xmax"] = x_max
    res_coord["ymin"] = y_min
    res_coord["ymax"] = y_max

    res_coord["xmin_s"] = x_s_min
    res_coord["xmax_s"] = x_s_max
    res_coord["ymin_s"] = y_s_min
    res_coord["ymax_s"] = y_s_max

    return res_coord


def draw_loc_on_Is(Is, dict_coord, is_same_min_loc):
    Is_cp = Is.copy()

    # min location.
    if is_same_min_loc:
        Is_cp = cv2.circle(
            Is_cp,
            (2 * dict_coord["xmin"], 2 * dict_coord["ymin"]),
            radius=1,
            color=(0, 0, 255),
            thickness=2,
        )
    else:
        Is_cp = cv2.circle(
            Is_cp,
            (2 * dict_coord["xmin"], 2 * dict_coord["ymin"]),
            radius=1,
            color=(255, 0, 0),
            thickness=1,
        )
        Is_cp = cv2.circle(
            Is_cp,
            (2 * dict_coord["xmin_s"], 2 * dict_coord["ymin_s"]),
            radius=2,
            color=(0, 255, 0),
            thickness=1,
        )

    # max location.
    Is_cp = cv2.rectangle(
        Is_cp,
        (2 * dict_coord["xmax"] - 1, 2 * dict_coord["ymax"] - 1),
        (2 * dict_coord["xmax"] + 1, 2 * dict_coord["ymax"] + 1),
        color=(255, 0, 0),
        thickness=1,
    )
    Is_cp = cv2.rectangle(
        Is_cp,
        (2 * dict_coord["xmax_s"] - 1, 2 * dict_coord["ymax_s"] - 1),
        (2 * dict_coord["xmax_s"] + 1, 2 * dict_coord["ymax_s"] + 1),
        color=(0, 255, 0),
        thickness=1,
    )

    return Is_cp


def draw_sampled_pixel_on_It(It, pix_tgt):
    It_cp = It.copy()
    It_cp[pix_tgt[0] - 1 : pix_tgt[0] + 1, pix_tgt[1] - 1 : pix_tgt[1] + 1, :] = [
        255,
        0,
        0,
    ]
    return It_cp


def get_n_instance(loader, n):
    it = iter(loader)
    data = []
    for i in range(n):
        data.append(next(it))
    return data


def get_an_instance_and_its_ray_sampler(data):
    bs_idx = np.random.randint(10)
    data = data[bs_idx]
    ray_sampler = RaySampler(data)
    # Get the list of the view to render.
    render_list = list(range(ray_sampler.render_imgs[0].shape[0]))
    render_list.remove(ray_sampler.src_view[0])
    render_view = int(np.random.choice(render_list, 1))  # int() for toy example test.

    return data, ray_sampler, render_view, bs_idx


def compute_metrics(It_pred, It, lpips_vgg, device):
    psnr = np.round(metrics.peak_signal_noise_ratio(It_pred, It, data_range=1), 2)
    ssim = metrics.structural_similarity(It_pred, It, multichannel=True, data_range=1)
    lpips = np.round(
        lpips_vgg(
            torch.from_numpy(It_pred)[None, ...].permute(0, 3, 1, 2).float().to(device),
            torch.from_numpy(It)[None, ...].permute(0, 3, 1, 2).float().to(device),
        ).item(),
        3,
    )

    return psnr, ssim, lpips


def get_n_colors(n):
    colors = [
        mcolors.to_rgba(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ]
    colors = colors * math.ceil(n / len(colors))
    return colors[:n]
