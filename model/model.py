import sys

sys.path.append("../")
import os
import torch
import pytorch_warmup as warmup

from network.hypernetwork import HyperNetworkSymmLocal
from utils.nerf_helpers import get_embedder
from network.feature_network import create_resnet_symm_local
from network.deeplabv3plus import deepLabv3Plus
from network.ray_transformer import RayTransformer
from network.resnet_mlp import PosEncodeResnet

def de_parallel(model):
    return model.module if hasattr(model, "module") else model

class HyperNeRFResNetSymmLocal(object):
    def __init__(self, args, ckpts_folder):
        super().__init__()
        self.args = args

        self.device = (
            torch.device("cuda:{}".format(args.local_rank))
            if args.local_rank >= 0
            else torch.device("cpu")
        )

        embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn      
        
        self.M = torch.tensor([[-1,0,0],[0,1,0],[0,0,1]]).float().to(self.device)
        
        # Type of feature concatenation that is performed. 
        self.feature_type = args.feature_type
        print(f'[Info] Feature type embedding: {self.feature_type}')
        
        # Model mode. 
        self.mode = 'train'
            
        ####################
        ## NeRF based model.
        ####################
        
        ## If Hypernetworks model are used. 
        if not args.discard_hypernetworks:
            print('[INFO] Using HyperNetworks to generate the weights of the NeRF. [INFO]')
            self.hypernetwork = HyperNetworkSymmLocal(
                hyper_in_ch=args.latent_dim,
                hyper_num_hidden_layers=args.hyper_num_hidden_layers,
                hyper_hidden_ch=args.latent_dim,
                hidden_ch=args.netwidth,
                num_hidden_layers=args.netdepth,
                num_local_layers=args.num_local_layers,
                input_ch=self.input_ch,
                input_ch_views=self.input_ch_views,
                local_feature_ch=args.local_feature_ch,
                outermost_linear=True,
                cosine_mod = args.cosine_mod,
                use_ray_transformer=args.use_ray_transformer
            ).to(self.device)

            if args.N_importance > 0:
                self.hypernetwork_fine = HyperNetworkSymmLocal(
                    hyper_in_ch=args.latent_dim,
                    hyper_num_hidden_layers=args.hyper_num_hidden_layers,
                    hyper_hidden_ch=args.latent_dim,
                    hidden_ch=args.netwidth,
                    num_hidden_layers=args.netdepth,
                    num_local_layers=args.num_local_layers,
                    input_ch=self.input_ch,
                    input_ch_views=self.input_ch_views,
                    local_feature_ch=args.local_feature_ch,
                    outermost_linear=True,
                    cosine_mod = args.cosine_mod,
                    use_ray_transformer=args.use_ray_transformer
                ).to(self.device)
            else:
                self.hypernetwork_fine = None
        
        ## Otherwise, use the ResNet MLP based NeRF model.
        else: 
            print('[INFO] Using ResNet MLP to generate the weights of the NeRF. [INFO]')
            pos_c = 3 
            in_c = 512 + 3 + 3 #2*256 for the position encoding, 3 for the view direction, 3 for the rgb value.
            args.mlp_feat_dim = 256
            args.mlp_block_num = 6
            self.coarse_net = PosEncodeResnet(args,pos_c,in_c,args.mlp_feat_dim,4,args.mlp_block_num).to(self.device)
            self.fine_net = PosEncodeResnet(args,pos_c,in_c,args.mlp_feat_dim,4,args.mlp_block_num).to(self.device)
            
        ###################
        ### Feature Network 
        ###################
        if not args.use_deepLabv3:
            self.feature_net = create_resnet_symm_local( arch="resnet34",
                                                    latent_dim=args.latent_dim,
                                                    pretrained=True,
                                                    index_interp=args.index_interp,
                                                    index_padding=args.index_padding,
                                                    upsample_interp=args.upsample_interp,
                                                    feature_scale=args.feature_scale,
                                                    use_first_pool=not args.no_first_pool,
                                                    use_first_layer_as_F=args.use_first_layer_as_F,
                                                    learnable_feature_upsampling=args.learnable_feature_upsampling,
                                                    ).to(self.device) 
        else:
            print('[INFO] Using DeepLabV3 as CNN feature extractor. [INFO]')
            self.feature_net = deepLabv3Plus(in_channels=3,out_channels=16*16, encoder_name='resnet34').to(self.device)
        
        
        # Cosine modulation that is not learned through the HyperNetwork model.
        if (args.cosine_mod['use'] and not args.cosine_mod['learn_through_hypernetwork']):
            self.cosine_linear = torch.nn.Linear(args.cosine_mod['G'],args.latent_dim).to(self.device)
            self.cosine_linear.apply(self.weights_init)
        
        ##################
        ## Ray Transformer
        ##################
        self.ray_transformer = RayTransformer(4,16,4,4).to(self.device) if args.use_ray_transformer else None
            
        #############################
        ### Optimizer & lr scheduler
        #############################
        
        ### 
        list_params_to_optimize = [
            {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
        ]
        if not args.discard_hypernetworks:
            list_params_to_optimize.append({"params": self.hypernetwork.parameters()})
            if self.hypernetwork_fine is not None:
                list_params_to_optimize.append({"params": self.hypernetwork_fine.parameters()})
        else: 
            list_params_to_optimize.append({"params": self.coarse_net.parameters()})
            list_params_to_optimize.append({"params": self.fine_net.parameters()})
            
        if (args.cosine_mod['use'] and not args.cosine_mod['learn_through_hypernetwork']):
            list_params_to_optimize.append({"params": self.cosine_linear.parameters()})
        
        if args.use_ray_transformer:
            print('RayTransformer are going optimized. ')
            list_params_to_optimize.append({"params": self.ray_transformer.parameters()})
            
        self.optimizer = torch.optim.AdamW(list_params_to_optimize, lr=args.lrate_mlp)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.N_iters)
        self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        self.start_step = self.load_from_ckpt(
            ckpts_folder,
            load_opt=not args.no_load_opt,
            load_scheduler=not args.no_load_scheduler,
        )

        if args.distributed:
            if not args.discard_hypernetworks:
                
                self.hypernetwork = torch.nn.parallel.DistributedDataParallel(
                    self.hypernetwork,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                )
                
                if self.hypernetwork_fine is not None:
                    self.hypernetwork_fine = torch.nn.parallel.DistributedDataParallel(
                        self.hypernetwork_fine,
                        device_ids=[args.local_rank],
                        output_device=args.local_rank,
                    )
            else:
                self.coarse_net = torch.nn.parallel.DistributedDataParallel(
                    self.coarse_net,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                ) 
                self.fine_net = torch.nn.parallel.DistributedDataParallel(
                    self.fine_net,
                    device_ids=[args.local_rank],
                    output_device = args.local_rank,
                )
                
            self.feature_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.feature_net
            )
            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
            )

            if self.cosine_linear is not None:
                self.cosine_linear = torch.nn.parallel.DistributedDataParallel(
                    self.cosine_linear,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                )
            if self.ray_transformer is not None:
                self.ray_transformer = torch.nn.parallel.DistributedDataParallel(
                    self.ray_transformer,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                )
                
            
            
            
    def get_symmetry_matrix(self):
        return self.M
    
    def encode(self, x):
        z = self.feature_net(x) 
        return z

    def switch_to_eval(self):
        self.feature_net.eval()
        if not self.args.discard_hypernetworks:
            self.hypernetwork.eval()
            self.hypernetwork_fine.eval()
        else:
            self.coarse_net.eval()
            self.fine_net.eval()
        if self.ray_transformer is not None: 
            self.ray_transformer.eval()
        self.mode = 'eval'

    def switch_to_train(self):
        self.feature_net.train()
        if not self.args.discard_hypernetworks:
            self.hypernetwork.train()
            self.hypernetwork_fine.train()
        else:
            self.coarse_net.train()
            self.fine_net.train()
        if self.ray_transformer is not None: 
            self.ray_transformer.train()    
        self.mode = 'train'
        
    def save_model(self, filename):
        to_save = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "warmup_scheduler": self.warmup_scheduler.state_dict(),
            
            "feature_net": de_parallel(self.feature_net).state_dict(),
        }

        if not self.args.discard_hypernetworks:
            to_save["hypernetwork"]: de_parallel(self.hypernetwork).state_dict()
            to_save["hypernetwork_fine"] = de_parallel(self.hypernetwork_fine).state_dict()
        else: 
            to_save['coarse_net'] = de_parallel(self.coarse_net).state_dict()
            to_save['fine_net'] = de_parallel(self.fine_net).state_dict()
            
        if self.ray_transformer is not None:
            to_save['ray_transformer'] = de_parallel(self.ray_transformer).state_dict()

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        to_load = torch.load(filename, map_location=self.device)
        print(f'Keys loaded: {to_load.keys()}')
        if load_opt:
            self.optimizer.load_state_dict(to_load["optimizer"])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])
            self.warmup_scheduler.load_state_dict(to_load["warmup_scheduler"])
     
        self.feature_net.load_state_dict(to_load["feature_net"])
        if not self.args.discard_hypernetworks:
            self.hypernetwork.load_state_dict(to_load["hypernetwork"]) # Strange behaviour on our deepV3_baseline.
            self.hypernetwork_fine.load_state_dict(to_load["hypernetwork_fine"])
        else:
            self.coarse_net.load_state_dict(to_load["coarse_net"])
            self.fine_net.load_state_dict(to_load["fine_net"])  
              
        if self.ray_transformer is not None and "ray_transformer" in to_load.keys():
            self.ray_transformer.load_state_dict(to_load["ray_transformer"])
            

    def load_from_ckpt(
        self, ckpts_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False
    ):
        """
        Load model from existing checkpoints and return the current step.

        :param ckpts_folder: the directory that stores ckpts
        :param load_opt:
        :param load_scheduler:
        :param force_latest_ckpt:
        :return: the current starting step
        """

        # All existing ckpts
        ckpts = []
        if os.path.exists(ckpts_folder):
            ckpts = [
                os.path.join(ckpts_folder, f)
                for f in sorted(os.listdir(ckpts_folder))
                if f.endswith(".pth")
            ]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print("[Info] Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("[Info] No ckpts found, training from scratch...")
            step = 0

        return step
    
    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)