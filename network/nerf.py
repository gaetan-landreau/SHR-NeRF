import sys

sys.path.append("../")
import torch
import torch.nn as nn
from utils.nerf_helpers import *

    
def run_nerf_symm_local(x,nerf_layers,input_ch,input_ch_views,local_feature_ch):
    
    
    input_pts, input_views, local_feature = torch.split(x,[input_ch,input_ch_views,local_feature_ch],dim=-1)
    
    # Local features layers. 
    locals = []
    for i in range(len(nerf_layers["local_linears"])):
        locals.append(nerf_layers["local_linears"][i](local_feature))
                
    h = nerf_layers["pts_linears"][0](input_pts)
    for i in range(1, len(nerf_layers["pts_linears"]) - 1):
        
            h = h + locals[i - 1]
                
    h = nerf_layers["pts_linears"][i](h)

    alpha = nerf_layers["alpha_linear"](h)
    h = nerf_layers["pts_linears"][-1](h)

    h = torch.cat([h, input_views], -1)
    h = nerf_layers["views_linear"](h)

    rgb = nerf_layers["rgb_linear"](h)
    outputs = torch.cat([rgb, alpha], -1)

    return outputs
