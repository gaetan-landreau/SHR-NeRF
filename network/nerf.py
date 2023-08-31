import sys

sys.path.append("../")
import torch
import torch.nn as nn
from utils.nerf_helpers import *

    
def run_nerf_symm_local(x,nerf_layers,input_ch,input_ch_views,local_feature_ch,cosine_mod):
    
    if cosine_mod['use']: 
        G = cosine_mod['G']
        input_pts, input_views, local_feature,cs = torch.split(x,[input_ch,input_ch_views,local_feature_ch,G],dim=-1)
    else: 
        input_pts, input_views, local_feature = torch.split(x,[input_ch,input_ch_views,local_feature_ch],dim=-1)
    
          
    # Local features layers. 
    locals = []
    for i in range(len(nerf_layers["local_linears"])):
        locals.append(nerf_layers["local_linears"][i](local_feature))
            
    cosine = nerf_layers["cosine_linear"](cs) if cosine_mod['use'] else None
    
    h = nerf_layers["pts_linears"][0](input_pts)
    for i in range(1, len(nerf_layers["pts_linears"]) - 1):
        if cosine_mod['use']:
            if cosine_mod['type_mod'] == 'local_feature_mod':
                h = h + locals[i - 1]*cosine
            if cosine_mod['type_mod'] == 'global_feature_mod':
                h = (h + locals[i - 1])*cosine
            if cosine_mod['type_mod'] == 'discard_local_feature':
                h = h*cosine
               
        else: 
            h = h + locals[i - 1]
                
        h = nerf_layers["pts_linears"][i](h)

    alpha = nerf_layers["alpha_linear"](h)
    h = nerf_layers["pts_linears"][-1](h)

    h = torch.cat([h, input_views], -1)
    h = nerf_layers["views_linear"](h)

    rgb = nerf_layers["rgb_linear"](h)
    outputs = torch.cat([rgb, alpha], -1)

    return outputs



    
