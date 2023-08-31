import sys 
sys.path.append('../')


import numpy as np 
import matplotlib.pyplot as plt 
import setproctitle
import tqdm 
from skimage.metrics import peak_signal_noise_ratio as PSNR


import torch 
from torch.utils.data import DataLoader

from datasets.srns_overfit import SRNsDatasetOverfit
from model.model import HyperNeRFResNetSymmLocal
from model.sample_ray import RaySampler
from model.render_ray import render_rays
from evaluation.eval_benchmark import pred_single_image
from opt import config_parser


parser = config_parser()
args = parser.parse_args()

device = torch.device("cuda:0")
args.archCNN = 'resnet34'

det = args.det
lindisp = args.lindisp
args.distributed = False
args.local_feature_ch = 512
args.local_rank = 0
args.eval_scene = 'cars'
args.eval_dataset = 'srns_dataset'
args.num_local_layers = 2
args.enforce_symmetry = {'status':False,
                         'on_coarse':False,'on_fine':False,
                         'concatenate_on_coarse':False, 'concatenate_on_fine':False}
args.no_load_opt = True
args.no_load_scheduler = True

setproctitle.setproctitle('[Gaetan - SymmNeRF Test]')


args.local_feature_ch = 1024
args.use_first_layer_as_F = False
args.no_first_pool = False
args.learnable_feature_upsampling = {'use':False,'type':None}
args.module_with_cosine = {"use":False,"G": 16}

model = HyperNeRFResNetSymmLocal(args,ckpts_folder = '')#/root/SymmNeRF-baseline/logs/srns_dataset/cars/baseline/ckpts/')

dataset = SRNsDatasetOverfit(args,mode='train',nb_iter=1000)

train_overfit_loader = DataLoader(dataset, batch_size= 1,#args.batch_size,
                              worker_init_fn=lambda _: np.random.seed(),
                              num_workers=args.workers,
                              pin_memory=False,
                              sampler=None,
                              shuffle=True)

# Create criterion
criterion = torch.nn.MSELoss()

I_target_list = []
I_source_list = []
I_pred_relu = []

for i,data in tqdm.tqdm(enumerate(train_overfit_loader)):
    
    ray_sampler = RaySampler(data)
    
    ray_batch = ray_sampler.random_sample(args.N_rand,device, use_bbox=True)

    z = model.encode(ray_batch['src_img'])

    ret = render_rays(ray_batch = ray_batch,
                          model = model,
                          device = device,
                          latent_vector = z,
                          enforceSymm = args.enforce_symmetry,
                          N_samples = args.N_samples,
                          module_with_cosine = args.module_with_cosine,
                          lindisp = args.lindisp,
                          N_importance = args.N_importance,
                          det = args.det,
                          raw_noise_std = args.raw_noise_std,
                          white_bkgd = args.white_bkgd)
    
    # Compute loss
    loss = criterion(ret['outputs_coarse']['rgb'], ray_batch['rgb']) + criterion(ret['outputs_fine']['rgb'], ray_batch['rgb'])

    loss.backward()
    model.optimizer.step()
    print(f'{loss}')
    
    '''
    if i % 100 == 0:
            

            render_list = list(range(ray_sampler.render_imgs[0].shape[0]))
            render_list.remove(ray_sampler.src_view[0])
            render_view = np.random.choice(render_list, 1)[0]

            gt_img = ray_sampler.render_imgs[0][render_view].permute(1, 2, 0)
            src_img = ray_sampler.render_imgs[0][ray_sampler.src_view[0]].permute(1,2,0)
            
            
            rgb_pred = pred_single_image(args,model,device,ray_sampler,render_view)
            
            It = gt_img.numpy()
            Is = src_img.numpy()
            Ifine = rgb_pred.numpy()
            
            I_target_list.append(It)
            I_source_list.append(Is)
            I_pred_relu.append(Ifine)
            
            psnr = PSNR(It,Ifine)
            print(f'Loss based: {loss}')
            print(f'PSNR based: {psnr:.3f}dB ')
    '''