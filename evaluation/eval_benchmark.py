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


def run_evaluation_benchmark(args,nb_test_instances,nb_view_per_instance):
    
    res_metrics = {'model_a':{'psnr':[],'ssim':[],'lpips':[]},
                'model_b':{'psnr':[],'ssim':[],'lpips':[]}
                }   
   
    # Data loading code
    test_dataset = dataset_dict[args.eval_dataset](args,'test',args.eval_scene)
    
    test_idx = np.random.randint(1,len(test_dataset),size=nb_test_instances)
    test_subset = Subset(test_dataset,test_idx)
    test_dataloader = DataLoader(test_subset,batch_size=1,shuffle=True,num_workers=4)
    
    device = torch.device(f'cuda:{args.local_rank}')
    
    # Baseline model loading. 
    args.cosine_mod = {'use':False,'learn_through_hypernetwork':False,'G':16}
    args.local_feature_ch = 1024
    args.use_deepLabv3 = False
    model_a = HyperNeRFResNetSymmLocal(args, ckpts_folder=args.ckpt_folder_a)
    
    # Cosine simmilarity enforced model loading. 
    args.cosine_mod = {'use':False,'learn_through_hypernetwork':False,'G':None}
    args.use_deepLabv3 = True
    args.local_feature_ch = 512
    model_b = HyperNeRFResNetSymmLocal(args, ckpts_folder=args.ckpt_folder_b)
    
    
    # LPIPS 
    lpips_vgg = lpips.LPIPS(net='vgg').to(device)
    
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
            args.cosine_mod = {'use':False,'learn_through_hypernetwork':False,'G':16}
            args.local_feature_ch = 1024            
            args.use_deepLabv3 = False
            Ipred_a = pred_single_image(args,model_a,device,ray_sampler,render_view)
            Ipred_a_np =Ipred_a.numpy()
        
            ###########
            ## Model B
            args.cosine_mod = {'use':False,'learn_through_hypernetwork':False,'G':None}
            args.local_feature_ch = 512     
            args.use_deepLabv3 = True
            Ipred_b = pred_single_image(args,model_b,device,ray_sampler,render_view)
            Ipred_b_np =Ipred_b.numpy()
            
            ##########################
            # Source and target images. 
            It = ray_sampler.render_imgs[0][render_view].permute(1, 2, 0).cpu()
            It_np = It.numpy()
            Is = ray_sampler.render_imgs[0][ray_sampler.src_view[0]].permute(1,2,0)
            Is_np = Is.numpy()
            
            ##########
            # Metrics.
            psnr_a = metrics.peak_signal_noise_ratio(Ipred_a_np,It_np)
            ssim_a= metrics.structural_similarity(Ipred_a_np,It_np,channel_axis = -1,data_range = 1)
            lpips_a = lpips_vgg(Ipred_a[None,...].permute(0,3,1,2).float().to(device),It[None,...].permute(0,3,1,2).float().to(device)).item()
        
            res_metrics['model_a']['psnr'].append(str(psnr_a))
            res_metrics['model_a']['ssim'].append(str(ssim_a))
            res_metrics['model_a']['lpips'].append(str(lpips_a))
            
            psnr_b = metrics.peak_signal_noise_ratio(Ipred_b_np,It_np)
            ssim_b= metrics.structural_similarity(Ipred_b_np,It_np,channel_axis = -1,data_range = 1)
            lpips_b = lpips_vgg(Ipred_b[None,...].permute(0,3,1,2).float().to(device),It[None,...].permute(0,3,1,2).float().to(device)).item()
            
            res_metrics['model_b']['psnr'].append(str(psnr_b))
            res_metrics['model_b']['ssim'].append(str(ssim_b))
            res_metrics['model_b']['lpips'].append(str(lpips_b))

            if i in idx_for_visuals:
                
                # Save visuals.
                img_stack = np.hstack([Is_np,It_np,Ipred_a_np,Ipred_b_np])
                cv2.imwrite(os.path.join(args.results_folder,'imgs',f'img_{i}_{render_view}.jpg'),img_stack*255)
                print('--> Saving done.')
                
        if i in subset_for_tmp_saving:
            # Save metrics results as json
            with open(os.path.join(args.results_folder,'results.json'),'w') as f:
                json.dump(res_metrics,f)
    
    # Save metrics results as json - final 
    with open(os.path.join(args.results_folder,'results.json'),'w') as f:
        json.dump(res_metrics,f)
    
    # Print results.
    data_model_a = transform_str2float(res_metrics['model_a'])
    data_model_b = transform_str2float(res_metrics['model_b'])
    
    psnr_a,psnr_b = np.mean(data_model_a['psnr']),np.mean(data_model_b['psnr'])
    ssim_a,ssim_b = np.mean(data_model_a['ssim']),np.mean(data_model_b['ssim'])
    lpips_a,lpips_b = np.mean(data_model_a['lpips']),np.mean(data_model_b['lpips'])
    
    # Get a print of the results.
    print(f'PSNR model A: {psnr_a}dB - PSNR model B: {psnr_b}dB \n')
    print(f'SSIM model A: {ssim_a} - SSIM model B: {ssim_b} \n')
    print(f'LPIPS model A: {lpips_a} - LPIPS model B: {lpips_b} \n')
    
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    
    args.local_rank = 0
    args.num_local_layers = 2
    args.no_load_opt = True
    args.no_load_scheduler = True
    
    # weights folders. 
    args.ckpt_folder_a = '/data/SymmNeRF-improved/logs/srns_dataset/cars/baseline_bis/ckpts/'
    args.ckpt_folder_b = '/data/SymmNeRF-improved/logs/srns_dataset/cars/baseline_deepV3/ckpts/'
    
   
    # logs saving results - folder creation. 
    args.results_folder = os.path.join('/data/SymmNeRF-improved/logs/evaluation',f'{args.expname}')
    os.makedirs(args.results_folder,exist_ok=True)
    os.makedirs(os.path.join(args.results_folder,'imgs'),exist_ok=True)
    
    setproctitle.setproctitle('[Gaetan] - Eval. Bench.')
    
    nb_test_instances=700
    nb_view_per_instance=10
    
    run_evaluation_benchmark(args,nb_test_instances,nb_view_per_instance)
    