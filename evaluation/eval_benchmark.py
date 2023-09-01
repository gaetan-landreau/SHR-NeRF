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

PSNR_HIGH_THRESHOLD = 100.

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
    model.switch_to_eval()
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

def run_evaluation_benchmark(args,nb_test_instances,nb_view_per_instance,models_info):
    
    # List all the model that are going to be evaluated. 
    models_name_list = sorted(list(models_info.keys()))
    
    # Save metrics. 
    res_metrics = dict([(model,{'psnr':[],'ssim':[],'lpips':[]}) for model in models_name_list])
    
    # Data loading code
    test_dataset = dataset_dict[args.eval_dataset](args,'test',args.eval_scene)
    
    test_idx = np.random.choice([i for i in range(len(test_dataset))],nb_test_instances,replace=False)
    test_subset = Subset(test_dataset,test_idx)
    test_dataloader = DataLoader(test_subset,batch_size=1,shuffle=True,num_workers=4)
    
    device = torch.device(f'cuda:{args.local_rank}')

    for model_name in models_name_list:
        # Get main args. 
        args.local_feature_ch = models_info[model_name]['local_feature_ch']
        args.use_deepLabv3 = models_info[model_name]['use_deepLabv3']
        args.latent_dim = models_info[model_name]['latent_dim']
        args.add_high_res_skip = models_info[model_name]['add_high_res_skip']
        
        # Model loading.
        print(f'[Info] - Loading model {model_name}')
        model = HyperNeRFResNetSymmLocal(args, ckpts_folder=models_info[model_name]['ckpts'])
        models_info[model_name]['model'] = model
    
    # LPIPS 
    lpips_vgg = lpips.LPIPS(net='vgg').to(device)
    
    # Visuals. 
    idx_for_visuals = np.random.randint(0,nb_test_instances,nb_test_instances//5 + 1)
    subset_for_tmp_saving = np.random.choice(idx_for_visuals,len(idx_for_visuals)//5 +1,replace=False)
   
    # Iterate over all test instances. 
    for i,test_data in enumerate(tqdm.tqdm(test_dataloader)):
        ray_sampler = RaySampler(test_data)
        
        render_list = list(range(ray_sampler.render_imgs[0].shape[0]))
        render_list.remove(ray_sampler.src_view[0])

        render_views = np.random.choice(render_list, nb_view_per_instance)

        # Iterate over all the target views to render. 
        for render_view in render_views: 
            
            ##########################
            # Source and target images. 
            It = ray_sampler.render_imgs[0][render_view].permute(1, 2, 0).cpu()
            It_np = It.numpy()
            Is = ray_sampler.render_imgs[0][ray_sampler.src_view[0]].permute(1,2,0)
            Is_np = Is.numpy()
            
            pred_imgs = [Is_np]
            # Inference for each model.
            for model_name in sorted(models_name_list): 
                
                args.local_feature_ch = models_info[model_name]['local_feature_ch']
                args.use_deepLabv3 = models_info[model_name]['use_deepLabv3']
                args.latent_dim = models_info[model_name]['latent_dim']
                args.add_high_res_skip = models_info[model_name]['add_high_res_skip']
                
                model_nvs = models_info[model_name]['model']
                
                Ipred = pred_single_image(args,model_nvs,device,ray_sampler,render_view)
                Ipred_np =Ipred.numpy()
        
                ##########
                # Metrics.
                psnr = metrics.peak_signal_noise_ratio(Ipred_np,It_np)
                ssim= metrics.structural_similarity(Ipred_np,It_np,channel_axis = -1,data_range = 1)
                _lpips = lpips_vgg(Ipred[None,...].permute(0,3,1,2).float().to(device),It[None,...].permute(0,3,1,2).float().to(device)).item()
                
                # Save index instances if psnr is anormaly too high / np.inf in a text file.
                if psnr > PSNR_HIGH_THRESHOLD:
                    f = open(os.path.join(args.results_folder,'anormaly_high_psnr.txt'),'a')
                    f.write(f'Instance {i} - View {render_view} - Model {model_name} - PSNR {psnr} \n')
                    f.close()
                    
                res_metrics[model_name]['psnr'].append(str(psnr))
                res_metrics[model_name]['ssim'].append(str(ssim))
                res_metrics[model_name]['lpips'].append(str(_lpips))
                pred_imgs.append(Ipred_np)
                
            if i in idx_for_visuals:
                pred_imgs.append(It_np)
                # Save visuals.
                img_stack = np.hstack(pred_imgs)
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
    for model_name in models_name_list:
        data_model = transform_str2float(res_metrics[model_name])
        psnr,ssim,_lpips = get_avg_scores(data_model)
        print(f' [RESULT] - Metrics for model {model_name}: PSNR: {psnr}dB - SSIM: {ssim} - LPIPS: {_lpips} \n')
   
def get_main_args(feature_ch,use_deepV3,latent_z_ch,add_high_res_skip,ckpt_folder):
        return {'local_feature_ch':feature_ch,
                'use_deepLabv3':use_deepV3,
                'latent_dim':latent_z_ch,
                'add_high_res_skip':add_high_res_skip,
                'ckpts':ckpt_folder}
        
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    
    args.local_rank = 1
    args.num_local_layers = 2
    args.no_load_opt = True
    args.no_load_scheduler = True
    args.src_view = '64'    # Easy setup with always the same source view. 
    
    # Main model information. 
    models_info = {'00_baseline': get_main_args(feature_ch=512,
                                            use_deepV3= False, 
                                            latent_z_ch = 256,
                                            add_high_res_skip = False,
                                            ckpt_folder='/data/baseline/ckpts/'),
                        
                   
                   
                   '01_ours':get_main_args(feature_ch = 576, 
                                            use_deepV3= True,
                                            latent_z_ch = 256,
                                            add_high_res_skip = True,
                                            ckpt_folder='/data/ours/ckpts/')
                   
                   
                   
                    }
    # logs saving results - folder creation. 
    args.results_folder = os.path.join('/data/logs/evaluation',f'{args.expname}')
    os.makedirs(args.results_folder,exist_ok=True)
    os.makedirs(os.path.join(args.results_folder,'imgs'),exist_ok=True)
    
    setproctitle.setproctitle(' Eval. Bench.')
    
    nb_test_instances=100
    nb_view_per_instance= 50
    
    run_evaluation_benchmark(args,nb_test_instances,nb_view_per_instance,models_info)
    
    