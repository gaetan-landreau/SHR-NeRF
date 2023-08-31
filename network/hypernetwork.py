import torch
import torch.nn as nn
import functools


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls

class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
    
class FCLinear(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features,out_features),)
    def forward(self,x):
        return self.net(x)
    
class FCBlock(nn.Module):
    def __init__(
        self,
        hidden_ch,
        num_hidden_layers,
        in_features,
        out_features,
        outermost_linear=False,
    ):
        super().__init__()
        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))

        if outermost_linear:
            self.net.append(nn.Linear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))

        self.net = nn.Sequential(*self.net)
        self.net.apply(self._init_weights)

    def __getitem__(self, item):
        return self.net[item]

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")

    def forward(self, x):
        return self.net(x)


class BatchLinear(nn.Module):
    def __init__(self, weights, biases):
        super().__init__()

        self.weights = weights
        self.biases = biases

    def __repr__(self):
        return "BatchLinear(in_ch=%d, out_ch=%d)" % (
            self.weights.shape[-1],
            self.weights.shape[-2],
        )

    def forward(self, x):
        output = x.matmul(
            self.weights.permute(
                *[i for i in range(len(self.weights.shape) - 2)], -1, -2
            )
        )
        output += self.biases
        return output


class HyperLinear(nn.Module):
    """A hypernetwork that predicts a single linear layer (weights & biases)."""

    def __init__(
        self, in_ch, out_ch, hyper_in_ch, hyper_num_hidden_layers, hyper_hidden_ch,cosine_mod=False
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        if not cosine_mod:
            self.hypo_params = FCBlock(
                in_features=hyper_in_ch,
                hidden_ch=hyper_hidden_ch,
                num_hidden_layers=hyper_num_hidden_layers,
                out_features=(in_ch * out_ch) + out_ch,
                outermost_linear=True,
            ) 
            self.hypo_params[-1].apply(self._init_last_hyperlayer)
        else: 
            self.hypo_params= FCLinear(in_features=hyper_in_ch, out_features=(in_ch * out_ch) + out_ch)
            self.hypo_params.apply(self._init_last_hyperlayer)
            
    #in_ch=16, out_ch=hidden_ch, hyper_in_ch=16, hyper_num_hidden_layers=1, hyper_hidden_ch=256
    
    def _init_last_hyperlayer(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")
            m.weight.data *= 1e-1

    def forward(self, hyper_input):
        hypo_params = self.hypo_params(hyper_input)

        # Indices explicit to catch erros in shape of output layer
        weights = hypo_params[..., : self.in_ch * self.out_ch]
        biases = hypo_params[
            ..., self.in_ch * self.out_ch : (self.in_ch * self.out_ch) + self.out_ch
        ]

        biases = biases.view(*(biases.size()[:-1]), 1, self.out_ch)
        weights = weights.view(*(weights.size()[:-1]), self.out_ch, self.in_ch)

        return BatchLinear(weights=weights, biases=biases)




class HyperLayer(nn.Module):
    """A hypernetwork that predicts a single Dense Layer, including LayerNorm and a ReLU."""

    def __init__(
        self, in_ch, out_ch, hyper_in_ch, hyper_num_hidden_layers, hyper_hidden_ch
    ):
        super().__init__()

        self.hyper_linear = HyperLinear(
            in_ch=in_ch,
            out_ch=out_ch,
            hyper_in_ch=hyper_in_ch,
            hyper_num_hidden_layers=hyper_num_hidden_layers,
            hyper_hidden_ch=hyper_hidden_ch,
        )
        self.norm_nl = nn.Sequential(
            nn.LayerNorm([out_ch], elementwise_affine=False), nn.ReLU(inplace=True)
        )

    def forward(self, hyper_input):
        """
        :param hyper_input: input to hypernetwork.
        :return: nn.Module; predicted fully connected network.
        """
        return nn.Sequential(self.hyper_linear(hyper_input), self.norm_nl)



class HyperNetworkSymmLocal(nn.Module):
    def __init__(
        self,
        hyper_in_ch,
        hyper_num_hidden_layers,
        hyper_hidden_ch,
        hidden_ch,
        num_hidden_layers,
        num_local_layers=3,
        input_ch=3,
        input_ch_views=3,
        local_feature_ch=1024,
        outermost_linear=False,
        cosine_mod = {},
        use_ray_transformer = False
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_local_layers = num_local_layers
        self.use_cosine_mod = cosine_mod['use'] and cosine_mod['learn_through_hypernetwork']
       
        if outermost_linear:
            PreconfHyperLinear = partialclass(
                HyperLinear,
                hyper_in_ch=hyper_in_ch,
                hyper_num_hidden_layers=hyper_num_hidden_layers,
                hyper_hidden_ch=hyper_hidden_ch,
                cosine_mod= False,
            )
        
        PreconfHyperLayer = partialclass(
            HyperLayer,
            hyper_in_ch=hyper_in_ch,
            hyper_num_hidden_layers=hyper_num_hidden_layers,
            hyper_hidden_ch=hyper_hidden_ch,
        )

        self.layers = nn.ModuleList()
                                          
        # local_linears
        for i in range(num_local_layers):
            self.layers.append(
                PreconfHyperLayer(in_ch=local_feature_ch, out_ch=hidden_ch)
            )

        # pts_linears
        self.layers.append(PreconfHyperLayer(in_ch=input_ch, out_ch=hidden_ch))
        for i in range(num_hidden_layers):
            self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=hidden_ch))
        # views_linear
        self.layers.append(
            PreconfHyperLayer(in_ch=hidden_ch + input_ch_views, out_ch=hidden_ch)
        )

        out_ch_alpha = 1 if not use_ray_transformer else 16
        if outermost_linear:
            # alpha_linear
            self.layers.append(PreconfHyperLinear(in_ch=hidden_ch, out_ch=out_ch_alpha))
            # rgb_linear
            self.layers.append(PreconfHyperLinear(in_ch=hidden_ch, out_ch=3))
        else:
            # alpha_linear
            self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=out_ch_alpha))
            # rgb_linear
            self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=3))
        
        if self.use_cosine_mod:
            self.layers.append(PreconfHyperLinear(in_ch=cosine_mod['G'], out_ch=hidden_ch,cosine_mod=True)) # 16 is the dimensionamity of the cosine similarity vector.
                                        
    def forward(self, hyper_input):
        net = []

        for i in range(len(self.layers)):
            net.append(self.layers[i](hyper_input))
        
        local_linears = net[: self.num_local_layers]
        pts_linears = net[
            self.num_local_layers : self.num_local_layers + self.num_hidden_layers + 1
        ]
        views_linear = net[-3] if not self.use_cosine_mod else net[-4]
        alpha_linear = net[-2]  if not self.use_cosine_mod else net[-3]
        rgb_linear = net[-1] if not self.use_cosine_mod else net[-2]
        cosine_linear = net[-1] if self.use_cosine_mod else None

        ret = {
            "local_linears": local_linears,
            "pts_linears": pts_linears,
            "views_linear": views_linear,
            "alpha_linear": alpha_linear,
            "rgb_linear": rgb_linear,
            "cosine_linear": cosine_linear,
        }

        return ret


if __name__ == "__main__":
    import torch

    z = torch.zeros([1, 256]).to("cuda:0")

    hypernetwork = HyperNetworkSymmLocal(
        hyper_in_ch=256,
        hyper_num_hidden_layers=1,
        hyper_hidden_ch=256,
        hidden_ch=256,
        num_hidden_layers=3,
        num_local_layers=3,
        input_ch=3,
        input_ch_views=3,
        local_feature_ch=1024,
        outermost_linear=True,
        cosine_mod=True
    ).to("cuda:0")

    ret = hypernetwork(z)
    #print(ret)
    print(ret['local_linears'])
    print('--'*10)
    for i in range(len(ret[('pts_linears')])):
        print(ret['pts_linears'][i])