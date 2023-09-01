import sys
sys.path.append('..')

from utils.general import * 
from model.render_ray import render_single_image
from torch.utils.data import DataLoader,Subset
from datasets import dataset_dict
from model.model import HyperNeRFResNetSymmLocal
from model.sample_ray import RaySampler
from opt import config_parser
import lpips 
    
import numpy as np 
from skimage import metrics
import setproctitle
import json 
import tqdm

def transform_str2float(data):
    for key in data.keys():
        data[key] = list(map(lambda x:float(x),data[key]))
    return data

def get_avg_scores(data):
    avg_psnr = np.mean(data['psnr'])
    avg_ssim = np.mean(data['ssim'])
    avg_lpips = np.mean(data['lpips'])
    return avg_psnr,avg_ssim,avg_lpips

def pred_single_image(args, model, device, ray_sampler, render_view):
    
    with torch.no_grad():
        ray_batch = ray_sampler.get_all_single_image(render_view, device)

        z = model.encode(ray_batch["src_img"])

        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            device=device,
            latent_vector=z,
            enforceSymm=args.enforce_symmetry,
            cosine_mod= args.cosine_mod,
            use_ray_transformer = args.use_ray_transformer,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            lindisp=args.lindisp,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=True,
        )

        rgb_pred = ret["outputs_fine"]["rgb"].detach().cpu()
        #model.switch_to_train()
        return rgb_pred


def run_rendering_views(args):
    
    # Data loading code
    test_dataset = dataset_dict[args.eval_dataset](args,'test',args.eval_scene)
    
    test_idx = np.random.randint(1,len(test_dataset),size=nb_test_instances)
    test_subset = Subset(test_dataset,test_idx)
    test_dataloader = DataLoader(test_subset,batch_size=1,shuffle=True,num_workers=4)
    
    device = torch.device(f'cuda:{args.local_rank}')
    
    # Baseline model loading. 
    args.local_feature_ch = 1024
    args.use_deepLabv3 = False
    model_a = HyperNeRFResNetSymmLocal(args, ckpts_folder=args.ckpt_folder_a)
    
    
    
    # Visuals. 
    idx_for_visuals = np.random.randint(0,nb_test_instances,nb_test_instances//5)
    subset_for_tmp_saving = np.random.choice(idx_for_visuals,len(idx_for_visuals)//5,replace=False)
    # Main evaluation loop. 
    for i,test_data in enumerate(tqdm.tqdm(test_dataloader)):
        ray_sampler = RaySampler(test_data)
        
        render_list = list(range(ray_sampler.render_imgs[0].shape[0]))
        render_list.remove(ray_sampler.src_view[0])

        render_views = np.random.choice(render_list, nb_view_per_instance)

        for render_view in render_views: 
        
            ###########
            ## Model A 
            args.local_feature_ch = 1024            
            args.use_deepLabv3 = False
            Ipred_a = pred_single_image(args,model_a,device,ray_sampler,render_view)
            Ipred_a_np =Ipred_a.numpy()
        
           
            
            ##########################
            # Source and target images. 
            It = ray_sampler.render_imgs[0][render_view].permute(1, 2, 0).cpu()
            It_np = It.numpy()
            Is = ray_sampler.render_imgs[0][ray_sampler.src_view[0]].permute(1,2,0)
            Is_np = Is.numpy()
            
            
            if i in idx_for_visuals:
                
                # Save visuals.
                img_stack = np.hstack([Is_np,It_np,Ipred_a_np,Ipred_b_np])
                cv2.imwrite(os.path.join(args.results_folder,'imgs',f'img_{i}_{render_view}.jpg'),img_stack*255)
                print('--> Saving done.')
                
        
    
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    
    args.local_rank = 0
    args.num_local_layers = 2
    args.no_load_opt = True
    args.no_load_scheduler = True
    
    # weights folders. 
    args.ckpt_folder = '/data/SymmNeRF-improved/logs/srns_dataset/cars/baseline_bis/ckpts/'
    
    # Source view used - Instance used.
    args.src_view = '64'
    args.specific_observation_idx = '420d1b7af7ceaad59ad3ae277a5ccc98'
    
    # logs saving results - folder creation. 
    args.results_folder = os.path.join('/data/SymmNeRF-improved/logs/evaluation',f'{args.expname}')
    os.makedirs(args.results_folder,exist_ok=True)
    os.makedirs(os.path.join(args.results_folder,'imgs'),exist_ok=True)
    
    setproctitle.setproctitle('[Gaetan] - Eval. Bench.')
    
    run_rendering_views(args)
    