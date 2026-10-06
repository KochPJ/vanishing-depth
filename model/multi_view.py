from torch import nn
import inspect
from torch import nn
import torch
import torch.nn.functional as F
import math
from model.pytransformers import TransformerEncoderDecoder, TransformerEncoderHead
from utils.stuff import load_fitting_state_dict


class FCMultiViewFusion(nn.Module):
    def __init__(self, channels: int, num_views: int, num_fusion_layers: int = 3):
        super().__init__()
        if num_fusion_layers > 1:
            c = channels * num_views
            step = int((c - channels) / num_fusion_layers)
            self.fusion = []
            for i in range(num_fusion_layers - 1):
                self.fusion.append(nn.Linear(c, c-step))
                c -= step

            self.fusion.append(nn.Linear(c, channels))
            self.fusion = nn.Sequential(*self.fusion)
        else:
            self.fusion = nn.Linear(channels * num_views, channels)

    def forward(self, x):
        x = x.flatten(1)
        return self.fusion(x)


class ConvMultiViewFusion(nn.Module):
    def __init__(self, num_views: int, num_fusion_layers: int = 3):
        super().__init__()
        if num_fusion_layers > 1:
            c = num_views
            step = int(num_views/ num_fusion_layers)
            self.fusion = []
            for i in range(num_fusion_layers-1):
                self.fusion.append(nn.Conv2d(c, c-step, kernel_size=1))
                c -= step

            self.fusion.append(nn.Conv2d(c, 1, kernel_size=1))
            self.fusion = nn.Sequential(*self.fusion)
        else:
            self.fusion = nn.Conv2d(num_views, 1, kernel_size=1)

    def forward(self, x):
        return self.fusion(x.unsqueeze(-1)).flatten(1)


class FCSqueezeAndExcitation(nn.Module):
    '''
    Inspired by:
    Copied from https://github.com/TUI-NICR/ESANet
    paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
    authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(FCSqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            activation,
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = self.fc(x)
        y = x * weighting
        return y


class MultiViewSqueezeAndExciteFusionAdd(nn.Module):
    '''
        Copied from https://github.com/TUI-NICR/ESANet
        paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
        authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channels, num_views: int, activation=nn.ReLU(inplace=True)):
        super(MultiViewSqueezeAndExciteFusionAdd, self).__init__()
        self.fuse = nn.ModuleList(
            [FCSqueezeAndExcitation(channels, activation=activation) for _ in range(num_views)])

    def forward(self, x):
        views = x.shape[1]
        out = []
        for i in range(views):
            out.append(self.fuse[i](x[:, i]))
        return sum(out)


class MultiViewSharedSqueezeAndExciteFusionAdd(nn.Module):
    '''
        Copied from https://github.com/TUI-NICR/ESANet
        paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
        authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channels, activation=nn.ReLU(inplace=True)):
        super(MultiViewSharedSqueezeAndExciteFusionAdd, self).__init__()
        self.fuse = FCSqueezeAndExcitation(channels, activation=activation)

    def forward(self, x):
        views = x.shape[1]
        out = []
        for i in range(views):
            out.append(self.fuse(x[:, i]))
        return sum(out)


class ConvDepthFusion(nn.Module):
    def __init__(self, channels: int, num_fusion_layers: int = 3):
        super().__init__()
        if num_fusion_layers > 1:
            c = channels * 2
            step = int(channels / num_fusion_layers)
            self.fusion = []
            for i in range(num_fusion_layers-1):
                self.fusion.append(nn.Conv2d(c, c-step, kernel_size=1))
                c -= step

            self.fusion.append(nn.Conv2d(c, channels, kernel_size=1))
            self.fusion = nn.Sequential(*self.fusion)
        else:
            self.fusion = nn.Conv2d(channels*2, channels, kernel_size=1)

    def forward(self, x, depth_x):
        return self.fusion(torch.cat((x, depth_x), dim=1))


class SqueezeAndExcitation(nn.Module):
    '''
    Copied from https://github.com/TUI-NICR/ESANet
    paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
    authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExciteFusionAdd(nn.Module):
    '''
        Copied from https://github.com/TUI-NICR/ESANet
        paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
        authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channels, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels, activation=activation)
        self.se_depth = SqueezeAndExcitation(channels, activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out


class MaxPool(nn.Module):
    def __init__(self, num_views: int):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=num_views)

    def __call__(self, x):
        return self.pool(x.permute(0, 2, 1)).squeeze(-1)


class AveragePool(nn.Module):
    def __init__(self, num_views: int):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=num_views)

    def __call__(self, x):
        return self.pool(x.permute(0, 2, 1)).squeeze(-1)


class Mean(nn.Module):
    def __init__(self, num_views: int):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=num_views)

    def __call__(self, x):
        return torch.mean(x, dim=1)



class TransformerMultiViewHead(nn.Module):
    def __init__(self, channels, num_views, layers=1, heads=8, with_positional_encoding=False, learnable_pe=False,
                 pc_embed_channels=64, pc_scale=200*math.pi, pc_temp=2000):
        super().__init__()
        self.num_views = num_views
        self.with_positional_encoding = with_positional_encoding
        self.learnable_pe = learnable_pe
        self.channels = channels
        self.pc_embed_channels = pc_embed_channels
        self.pc_temp = pc_temp
        self.pc_scale = pc_scale
        self.tf = Transformer(channels, layers, heads, channels, channels)
        #self.tf = TransformerEncoderHead(d_model=channels, nhead=heads, num_encoder_layers=layers)
        if self.with_positional_encoding:
            if self.learnable_pe:
                self.pos_emb = nn.Embedding(self.num_views, self.channels)
            else:
                self.pos_emb = torch.zeros((self.num_views, self.channels))
                self.pos_emb[:, :self.pc_embed_channels] = get_pos_embed(torch.arange(0, self.num_views).unsqueeze(0),
                                                                         pc_embed_channels,
                                                                         scale=self.pc_scale,
                                                                         temperature=self.pc_temp).squeeze(0)
        else:
            self.pos_emb = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.with_positional_encoding and self.learnable_pe:
            nn.init.uniform_(self.pos_emb.weight)

    def __call__(self, x):
        if self.pos_emb is not None:
            if self.learnable_pe:
                pos_embed = self.pos_emb.weight
            else:
                pos_embed = self.pos_emb.to(x.device)
            x = x + pos_embed
        return self.tf(x)


class TransformerMultiViewHeadDecoder(nn.Module):
    def __init__(self, channels, num_views, layers=1, heads=8, with_positional_encoding=False, learnable_pe=False,
                 pc_embed_channels=16, pc_scale=200*math.pi, pc_temp=2000):
        super().__init__()
        self.num_views = num_views + 1
        self.with_positional_encoding = with_positional_encoding
        self.learnable_pe = learnable_pe
        self.channels = channels
        self.pc_embed_channels = pc_embed_channels
        self.pc_scale = pc_scale
        self.pc_temp = pc_temp
        #self.tf = Transformer(channels, layers, heads, channels, channels)
        self.tf = TransformerEncoderHead(d_model=channels, nhead=heads, num_encoder_layers=layers)
        if self.with_positional_encoding:
            if self.learnable_pe:
                self.pos_emb = nn.Embedding(self.num_views, self.channels)
            else:
                self.pos_emb = torch.zeros((self.num_views, self.channels))
                self.pos_emb[:, :self.pc_embed_channels] = get_pos_embed(torch.arange(0, self.num_views).unsqueeze(0),
                                                                         pc_embed_channels,
                                                                         scale=self.pc_scale,
                                                                         temperature=self.pc_temp).squeeze(0)
        else:
            self.pos_emb = None

        self.query_emb = nn.Embedding(1, self.channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.with_positional_encoding and self.learnable_pe:
            nn.init.uniform_(self.pos_emb.weight)

    def __call__(self, x, weight=None):
        query = self.query_emb.weight.repeat(len(x), 1)
        if weight is not None:
            weight_ = torch.zeros((len(x), self.channels))
            weight_[:, :self.pc_embed_channels] = get_pos_embed(weight, self.pc_embed_channels,
                                                                scale=self.pc_scale,
                                                                temperature=self.pc_temp).squeeze(1)
            query = query + weight_.to(query.device)

        x = torch.hstack([query.unsqueeze(1), x])
        if self.pos_emb is not None:
            if self.learnable_pe:
                pos_embed = self.pos_emb.weight
            else:
                pos_embed = self.pos_emb.to(x.device)
            x = x + pos_embed
        return self.tf(x)[:, 0]



class WeightNet(nn.Module):
    def __init__(self, num_classes: int, out_channels: int, pc_embed_channels: int = 64, pc_scale=200*math.pi,
                 with_fc=True, pc_temp=2000):
        super(WeightNet, self).__init__()
        self.weight_fusion = nn.Sequential(
            nn.Linear(pc_embed_channels, pc_embed_channels * 2),
            nn.Linear(pc_embed_channels * 2, pc_embed_channels * 4),
            nn.Linear(pc_embed_channels * 4, pc_embed_channels * 8),
            nn.Linear(pc_embed_channels * 8, pc_embed_channels * 8),
            nn.Linear(pc_embed_channels * 8, out_channels),
        )

        self.with_fc = with_fc
        if self.with_fc:
            self.fc = nn.Linear(out_channels, num_classes)
            self.drop = nn.Dropout(0.5)
            self.out_channels = num_classes
        else:
            self.drop = None
            self.fc = None
            self.out_channels = out_channels

        self.pc_embed_channels = pc_embed_channels
        self.pc_scale = pc_scale
        self.pc_temp = pc_temp

    def forward(self, weight):
        weight = get_pos_embed(weight, self.pc_embed_channels, scale=self.pc_scale, temperature=self.pc_temp).squeeze(1)
        weight = self.weight_fusion(weight)
        if self.fc is not None:
            if self.training:
                weight = self.drop(weight)
            weight = self.fc(weight)
        return weight



class PropertyNet(nn.Module):
    def __init__(self, num_classes: int, out_channels: int, pc_embed_channels: int = 64,  num_properties: int = 4,
                 pc_scale=200*math.pi, with_fc=True, pc_temp=2000):
        super(PropertyNet, self).__init__()
        self.num_properties = num_properties

        self.in_channels = num_properties*pc_embed_channels
        #print(self.num_properties, 'num_properties')
        #print(self.in_channels, 'in_channels')

        self.property_fusion = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels * 2),
            nn.Linear(self.in_channels * 2, self.in_channels * 2),
            nn.Linear(self.in_channels * 2, self.in_channels * 2),
            nn.Linear(self.in_channels * 2, out_channels),
            nn.Linear(out_channels, out_channels),
        )

        self.with_fc = with_fc
        if self.with_fc:
            self.fc = nn.Linear(out_channels, num_classes)
            self.drop = nn.Dropout(0.5)
            self.out_channels = num_classes
        else:
            self.drop = None
            self.fc = None
            self.out_channels = out_channels

        self.pc_embed_channels = pc_embed_channels
        self.pc_scale = pc_scale
        self.pc_temp = pc_temp

    def forward(self, property):
        #print(property.shape, 1)
        property = get_pos_embed(property, self.pc_embed_channels, scale=self.pc_scale,
                                 temperature=self.pc_temp).squeeze(1)
        #print(property.shape, 2)
        #property = property.view(len(property), -1)
        property = property.flatten(1)

        #print(property.shape, 2)
        property = self.property_fusion(property)
        if self.fc is not None:
            if self.training:
                property = self.drop(property)
            property = self.fc(property)
        return property


class TransfomerEncoderDecoderMultiViewHead(nn.Module):
    def __init__(self, channels, num_views, layers=1, heads=8, dim_feedforward=2048, activation="relu",
                 normalize_before=False, return_intermediate_dec=False, with_positional_encoding=True,
                 pc_embed_channels=64, learnable_pe=False, use_weightnet=False, use_propertyNet=False,
                 pc_temp=2000, pc_scale=200*math.pi,
                 freeze_propertynet=False):
        super().__init__()
        self.tf = TransformerEncoderDecoder(d_model=channels, nhead=heads, num_encoder_layers=layers,
                                            num_decoder_layers=layers, dim_feedforward=dim_feedforward,
                                            activation=activation, normalize_before=normalize_before,
                                            return_intermediate_dec=return_intermediate_dec)
        self.query_embed = nn.Embedding(1, channels)

        self.with_positional_encoding = with_positional_encoding
        self.learnable_pe = learnable_pe
        self.pc_embed_channels = pc_embed_channels
        self.num_views = num_views
        self.channels = channels
        self.pc_scale = pc_scale
        self.pc_temp = pc_temp
        if self.with_positional_encoding:
            if self.learnable_pe:
                self.pos_emb = nn.Embedding(self.num_views, self.channels)
            else:
                self.pos_emb = torch.zeros((self.num_views, self.channels))
                self.pos_emb[:, :self.pc_embed_channels] = get_pos_embed(torch.arange(0, self.num_views).unsqueeze(0),
                                                                         pc_embed_channels,
                                                                         scale=self.pc_scale,
                                                                         temperature=self.pc_temp).squeeze(0)

        else:
            self.pos_emb = None

        self.contionalizer = None
        self.use_weightnet = use_weightnet
        self.use_propertyNet = use_propertyNet
        self.freeze_propertynet = freeze_propertynet
        print('here', use_weightnet, use_propertyNet)
        if self.use_weightnet or self.use_propertyNet:

            if use_weightnet:
                weightet_path = '/home/kochpaul/git/multi-view-part-recognition/results/run100/' \
                                'WeightNet10000/WeightNet10000_best.ckpt'
                sd = torch.load(weightet_path, map_location='cpu')['state_dict']
                self.contionalizer = WeightNet(0, channels, with_fc=False, pc_scale=self.pc_scale, pc_temp=self.pc_temp,
                                       pc_embed_channels=self.pc_embed_channels)
                self.contionalizer = load_fitting_state_dict(self.contionalizer, sd)
                print('loaded pretrained weightnet')

            elif use_propertyNet:
                #propertynet_path = '/home/kochpaul/git/multi-view-part-recognition/results/run100/' \
                #                   'PropertyNet/PropertyNet_best.ckpt'
                propertynet_path = '/home/kochpaul/git/multi-view-part-recognition/results/run100/' \
                                   'PropertyNet_with_random/PropertyNet_with_random_best.ckpt'
                sd = torch.load(propertynet_path, map_location='cpu')['state_dict']
                self.contionalizer = PropertyNet(0, channels, with_fc=False, pc_scale=self.pc_scale, pc_temp=self.pc_temp,
                                               pc_embed_channels=self.pc_embed_channels)
                self.contionalizer = load_fitting_state_dict(self.contionalizer, sd)
                print('loaded pretrained propertynet')

            if self.freeze_propertynet:
                for param in self.contionalizer.parameters():
                    param.requires_grad = False
                #for param in self.weightNet.parameters():
                #    print(param.requires_grad)

            self.query_emb = None
        else:
            self.query_emb = nn.Embedding(1, self.channels)
        self.reset_parameters()

    def reset_parameters(self):
        if self.with_positional_encoding and self.learnable_pe:
            nn.init.uniform_(self.pos_emb.weight)

    def __call__(self, x, conditions=None):
        #print(x.shape, conditions.shape)

        if self.pos_emb is not None:
            if self.learnable_pe:
                pos_embed = self.pos_emb.weight
            else:
                pos_embed = self.pos_emb.to(x.device)
        else:
            pos_embed = None

        if conditions is not None:
            if self.contionalizer is not None:
                query = self.contionalizer(conditions)
                #print('querry from weight net')
            else:
                #print(weight)
                conditions = get_pos_embed(conditions, self.pc_embed_channels,
                                       scale=self.pc_scale, temperature=self.pc_temp).flatten(1).squeeze(1)
                #print(conditions.shape)
                query = self.query_embed.weight.repeat(len(x), 1)
                #print(query.shape)
                query[:, :conditions.shape[-1]] += conditions
                #print('weight added to querry')
        else:
            query = self.query_embed.weight.repeat(len(x), 1)
            #print('querry pure')

        out, _ = self.tf(x, None, query, pos_embed)
        return out.squeeze(1)

#def get_pos_embed(x, num_pos_feats=64, temperature=2000, scale=200*math.pi):
def get_pos_embed(x, num_pos_feats=16, temperature=2000, scale=100*math.pi):
    if scale is None:
        scale = 2 * math.pi
        x = x * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = (2 * torch.div(dim_t, 2, rounding_mode='trunc')) / num_pos_feats
    dim_t = temperature ** dim_t  # dim_t // 2
    dim_t = dim_t.to(x.device)
    x = x[:, :, None] / dim_t
    x = torch.stack((x[:, :, 0::2].sin(),
                     x[:, :, 1::2].cos()), dim=3).flatten(2)
    return x


__fusion__ = {
    'Squeeze&Excite': MultiViewSqueezeAndExciteFusionAdd,
    'SharedSqueeze&Excite': MultiViewSharedSqueezeAndExciteFusionAdd,
    'FC': FCMultiViewFusion,
    'Conv': ConvMultiViewFusion,
    'Transformer': TransformerMultiViewHead,
    'max-pool': MaxPool,
    'average-pool': AveragePool,
    'mean': Mean,
    'TransfomerEncoderDecoderMultiViewHead': TransfomerEncoderDecoderMultiViewHead,
    'TransformerMultiViewHeadDecoder': TransformerMultiViewHeadDecoder
}


def get_multiview_model(args, model, use_n_blocks=4, use_avepool=True):
    if args.multiview:
        f = __fusion__.get(args.fusion)
        if f is None:
            raise ValueError('Fusion "" does not exist, {} are implemented'.format(args.fusion, __fusion__.keys()))
        num_views = len(args.views.split('-'))
        if not args.rgb_only:
            model = MultiViewRGBD(model, args.num_classes, num_views, f, freeze_encoder=args.freeze_encoder,
                                  dropout=args.dropout_linear, use_n_blocks=use_n_blocks, use_avepool=use_avepool,
                                  lrs=args.lr, split_view=args.split_view, split_modalities=args.split_modalities
                                  )
        else:
            model = MultiView(model, args.num_classes, num_views, f, freeze_encoder=args.freeze_encoder,
                              dropout=args.dropout_linear, use_n_blocks=use_n_blocks, use_avepool=use_avepool,
                              lrs=args.lr, split_view=args.split_view)
    return model


def get_singleview_model(args, model, omni=False, use_n_blocks=4, use_avepool=True):



    if not args.rgb_only:
        model = SingleViewRGBD(model, args.num_classes, freeze_encoder=args.freeze_encoder,
                               dropout=args.dropout_linear, lrs=args.lr, omni=omni, use_n_blocks=use_n_blocks, use_avepool=use_avepool)
    else:
        model = SingleView(model, args.num_classes, freeze_encoder=args.freeze_encoder,
                           dropout=args.dropout_linear, lrs=args.lr, omni=omni, use_n_blocks=use_n_blocks, use_avepool=use_avepool)
    return model


class MultiView(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, num_views: int, fusion: nn.Module, dropout=0.5,
                 with_positional_encoding=True, learnable_pe=True,
                 pc_embed_channels=64, pc_temp=2000, pc_scale=200 * math.pi, freeze_encoder=True, use_n_blocks=4,
                 use_avepool=True, lrs=1, split_view=False):
        super(MultiView, self).__init__()
        self.encoder = encoder

        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            print('Freezing Encoder')
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.use_n_blocks = use_n_blocks
        self.use_avepool = use_avepool
        self.in_channels = encoder.out_channels * self.use_n_blocks
        self.split_view = split_view
        if self.use_avepool and not self.split_view:
            self.in_channels += encoder.out_channels
        elif self.split_view:
            if self.use_avepool:
                n = num_views
            else:
                n = 0
            self.in_channels = encoder.out_channels
            num_views = num_views * self.use_n_blocks + n
        self.num_views = num_views
        print('num_views', num_views)

        args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
        arg_dict = {'channels': self.in_channels,
                    'num_views': num_views,
                    'with_positional_encoding': with_positional_encoding,
                    'learnable_pe': learnable_pe,
                    'pc_embed_channels': pc_embed_channels,
                    'pc_temp': pc_temp,
                    'pc_scale': pc_scale}

        args = {arg: arg_dict[arg] for arg in args if arg in arg_dict}
        self.multiViewFusion = fusion(**args)

        n_classifiers = 1
        if isinstance(lrs, list):
            n_classifiers = len(lrs)
            if not isinstance(dropout, list):
                dropout = [dropout]
        if isinstance(dropout, list):
            n_classifiers = n_classifiers * len(dropout)
            if not isinstance(lrs, list):
                lrs = [lrs]

        self.n_classifiers = n_classifiers
        self.lrs = lrs
        if n_classifiers == 1:
            self.fc = nn.Linear(self.in_channels, num_classes)
            self.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.fc.bias.data.zero_()
            self.drop_out = nn.Dropout(p=dropout)
        else:
            modules = []
            dropouts = []
            self.lrs = []
            for drop in dropout:
                for lr in lrs:
                    modules.append(nn.Linear(self.in_channels, num_classes))
                    modules[-1].weight.data.normal_(mean=0.0, std=0.01)
                    modules[-1].bias.data.zero_()
                    dropouts.append(nn.Dropout(p=drop))
                    self.lrs.append(lr)
            self.fc = nn.ModuleList(modules)
            self.drop_out = nn.ModuleList(dropouts)


    def forward(self, x):
        bs, i, c, h, w = x.shape
        x = x.view(bs * i, c, h, w)

        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)

        if isinstance(x, dict):
            x = x['return']

        if isinstance(x, list):
            x = create_linear_input(x[:-1], self.use_n_blocks, self.use_avepool)
        # encoder regularisation

        # get shape back again and flatten for each set
        x = x.view(bs, i, -1)
        if self.split_view:
            x = x.view(bs, -1, self.in_channels)

        # fuse
        x = self.multiViewFusion(x)

        #print(type(x), x.shape, x.device)

        # classify
        if self.n_classifiers > 1:
            if self.training:
                x = [fc(drop(x)) for fc, drop in zip(self.fc, self.drop_out)]
            else:
                x = [fc(x) for fc in self.fc]
        else:
            if self.training:
                x = self.drop_out(x)

            x = self.fc(x)
        return x


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool, stack=False):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    if stack:
        output = torch.cat([tokens[:, 0].unsqueeze(1) for tokens in intermediate_output], dim=1)
    else:
        output = torch.cat([tokens[:, 0] for tokens in intermediate_output], dim=-1)

    if use_avgpool:
        if stack:
            output = torch.cat(
                (
                    output,
                    torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(1)  # patch tokens
                ),
                dim=1,
            )
        else:
            output = torch.cat(
                (
                    output,
                    torch.mean(intermediate_output[-1][:, 1:], dim=1),  # patch tokens
                ),
                dim=-1,
            )
            output = output.reshape(output.shape[0], -1)

    return output.float()



class SingleView(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, dropout=0.0, freeze_encoder=True,
                 use_n_blocks=4, use_avepool=True, lrs=1, omni=False):
        super(SingleView, self).__init__()
        self.encoder = encoder
        self.use_n_blocks = use_n_blocks
        self.use_avepool = use_avepool
        self.omni = omni

        self.freeze_encoder = freeze_encoder

        self.in_channels = encoder.out_channels * self.use_n_blocks
        if self.use_avepool:
            self.in_channels += encoder.out_channels


        if self.freeze_encoder:
            print('Freezing Encoder')
            for param in self.encoder.parameters():
                param.requires_grad = False

        n_classifiers = 1
        if isinstance(lrs, list):
            n_classifiers = len(lrs)
            if not isinstance(dropout, list):
                dropout = [dropout]
        if isinstance(dropout, list):
            n_classifiers = n_classifiers * len(dropout)
            if not isinstance(lrs, list):
                lrs = [lrs]

        self.n_classifiers = n_classifiers
        self.lrs = lrs
        if n_classifiers == 1:
            self.fc = nn.Linear(self.in_channels, num_classes)
            self.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.fc.bias.data.zero_()
            self.drop_out = nn.Dropout(p=dropout)
        else:
            modules = []
            dropouts = []
            self.lrs = []
            for drop in dropout:
                for lr in lrs:
                    modules.append(nn.Linear(self.in_channels, num_classes))
                    modules[-1].weight.data.normal_(mean=0.0, std=0.01)
                    modules[-1].bias.data.zero_()
                    dropouts.append(nn.Dropout(p=drop))
                    self.lrs.append(lr)

            self.fc = nn.ModuleList(modules)
            self.drop_out = nn.ModuleList(dropouts)

        #cp = torch.load('/home/kochpaul/Downloads/dinov2_vits14_linear4_head.pth')
        #self.fc.load_state_dict(cp)
        #print(cp['weight'].shape)
        #input()

    def forward(self, x):

        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)

        if isinstance(x, dict):
            x = x['return']

        if isinstance(x, list):
            if self.omni:
                #x_ = [x[-1]]
                #for xemb in x[1:-1]:
                #    x_.append(torch.mean(xemb.flatten(2), 2))
                #    print(x_[-1].shape, xemb.shape)
                #x = torch.cat(x_, dim=1)
                x = x[-1]
            else:
                x = create_linear_input(x[:-1], self.use_n_blocks, self.use_avepool)

        # classify
        if self.n_classifiers > 1:
            if self.training:
                x = [fc(drop(x)) for fc, drop in zip(self.fc, self.drop_out)]
            else:
                x = [fc(x) for fc in self.fc]
        else:
            # encoder regularisation
            if self.training:
                x = self.drop_out(x)

            x = self.fc(x)
        return x


class MultiViewRGBD(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, num_views: int, fusion: nn.Module, dropout: float = 0.5,
                 with_positional_encoding=True, learnable_pe=True,
                 pc_embed_channels=64, pc_temp=2000, pc_scale=200*math.pi, freeze_encoder=True,
                 freeze_color_encoder=True, use_n_blocks=4, use_avepool=True, lrs=1, split_view=False,
                 split_modalities=False):
        super(MultiViewRGBD, self).__init__()
        self.generator = None
        if isinstance(encoder, tuple):
            encoder, self.generator = encoder
            print('Freezing Generator')
            for param in self.generator.parameters():
                param.requires_grad = False

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.freeze_color_encoder = freeze_color_encoder
        self.use_n_blocks = use_n_blocks
        self.use_avepool = use_avepool
        self.split_view = split_view
        self.split_modalities = split_modalities

        if self.freeze_encoder:
            print('Freezing Encoder')
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif self.freeze_color_encoder:
            print('Freezing Only Color Encoder')
            for param in self.encoder.color_encoder.parameters():
                param.requires_grad = False

        self.in_channels = encoder.out_channels * self.use_n_blocks
        if self.use_avepool and not self.split_view:
            self.in_channels += encoder.out_channels
        elif self.split_view:
            if self.use_avepool:
                n = num_views
            else:
                n = 0

            if encoder.cat_outs and self.split_modalities:
                self.in_channels = encoder.out_channels // 2
                num_views = num_views * 2
                n = n * 2
            else:
                self.in_channels = encoder.out_channels

            num_views = num_views * self.use_n_blocks + n

        self.num_views = num_views
        print('num_views', num_views)

        args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
        arg_dict = {'channels': self.in_channels,
                    'num_views': num_views,
                    'with_positional_encoding': with_positional_encoding,
                    'learnable_pe': learnable_pe,
                    'pc_embed_channels': pc_embed_channels,
                    'pc_temp': pc_temp,
                    'pc_scale': pc_scale}

        args = {arg: arg_dict[arg] for arg in args if arg in arg_dict}
        self.multiViewFusion = fusion(**args)

        n_classifiers = 1
        if isinstance(lrs, list):
            n_classifiers = len(lrs)
            if not isinstance(dropout, list):
                dropout = [dropout]
        if isinstance(dropout, list):
            n_classifiers = n_classifiers * len(dropout)
            if not isinstance(lrs, list):
                lrs = [lrs]

        self.n_classifiers = n_classifiers
        self.lrs = lrs
        if n_classifiers == 1:
            self.fc = nn.Linear(self.in_channels, num_classes)
            self.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.fc.bias.data.zero_()
            self.drop_out = nn.Dropout(p=dropout)
        else:
            modules = []
            dropouts = []
            self.lrs = []
            for drop in dropout:
                for lr in lrs:
                    modules.append(nn.Linear(self.in_channels, num_classes))
                    modules[-1].weight.data.normal_(mean=0.0, std=0.01)
                    modules[-1].bias.data.zero_()
                    dropouts.append(nn.Dropout(p=drop))
                    self.lrs.append(lr)
            self.fc = nn.ModuleList(modules)
            self.drop_out = nn.ModuleList(dropouts)

    def forward(self, x, depth=None):
        #print('depth mv input type, {}'.format(type(depth)))
        bs, i, c, h, w = x.shape
        x = x.view(bs * i, c, h, w)

        if self.generator is not None:
            depth = self.generator(x, depth)
        else:
            bs, i, c, h, w = depth.shape
            depth = depth.view(bs * i, c, h, w)

        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                #print(x.shape, depth.shape)
                x = self.encoder(x, depth)
        else:
            x = self.encoder(x, depth)

        if isinstance(x, dict):
            x = x['return']

        if isinstance(x, list):
            x = create_linear_input(x[:-1], self.use_n_blocks, self.use_avepool)


        # get shape back again and flatten for each set
        #print(x.shape, self.num_views, self.in_channels)
        #print(x.shape, self.num_views, self.in_channels)
        #x = x.view(bs, -1, self.in_channels)

        x = x.view(bs, i, -1)
        if self.split_view:
            x = x.view(bs, -1, self.in_channels)
        #print(x.shape)

        # fuse
        x = self.multiViewFusion(x)

        # classify
        if self.n_classifiers > 1:
            if self.training:
                x = [fc(drop(x)) for fc, drop in zip(self.fc, self.drop_out)]
            else:
                x = [fc(x) for fc in self.fc]
        else:
            # encoder regularisation
            if self.training:
                x = self.drop_out(x)

            x = self.fc(x)

        return x


class SingleViewRGBD(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, dropout: float = 0.5, freeze_encoder=True,
                 freeze_color_encoder=True, use_n_blocks=4, use_avepool=True, lrs=1, omni=False):
        super(SingleViewRGBD, self).__init__()
        self.generator = None
        if isinstance(encoder, tuple):
            encoder, self.generator = encoder
            print('Freezing Generator')
            for param in self.generator.parameters():
                param.requires_grad = False

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.freeze_color_encoder = freeze_color_encoder
        self.omni = omni

        if self.freeze_encoder:
            print('Freezing Encoder')
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif self.freeze_color_encoder:
            print('Freezing Only Color Encoder')
            for param in self.encoder.color_encoder.parameters():
                param.requires_grad = False

        self.use_n_blocks = use_n_blocks
        self.use_avepool = use_avepool
        self.in_channels = encoder.out_channels * self.use_n_blocks
        if self.use_avepool:
            self.in_channels += encoder.out_channels

        n_classifiers = 1
        if isinstance(lrs, list):
            n_classifiers = len(lrs)
            if not isinstance(dropout, list):
                dropout = [dropout]
        if isinstance(dropout, list):
            n_classifiers = n_classifiers * len(dropout)
            if not isinstance(lrs, list):
                lrs = [lrs]

        self.n_classifiers = n_classifiers
        self.lrs = lrs
        if n_classifiers == 1:
            self.fc = nn.Linear(self.in_channels, num_classes)
            self.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.fc.bias.data.zero_()
            self.drop_out = nn.Dropout(p=dropout)
        else:
            modules = []
            dropouts = []
            self.lrs = []
            for drop in dropout:
                for lr in lrs:
                    modules.append(nn.Linear(self.in_channels, num_classes))
                    modules[-1].weight.data.normal_(mean=0.0, std=0.01)
                    modules[-1].bias.data.zero_()
                    dropouts.append(nn.Dropout(p=drop))
                    self.lrs.append(lr)
            self.fc = nn.ModuleList(modules)
            self.drop_out = nn.ModuleList(dropouts)


    def forward(self, x, depth=None):
        #print('depth mv input type, {}'.format(type(depth)))

        if self.generator is not None:
            depth = self.generator(x, depth)

        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x, depth)
        else:
            x = self.encoder(x, depth)

        if isinstance(x, list):
            if self.omni:
                x = x[-1]
            else:
                x = create_linear_input(x[:-1], self.use_n_blocks, self.use_avepool)

        # classify
        if self.n_classifiers > 1:
            if self.training:
                x = [fc(drop(x)) for fc, drop in zip(self.fc, self.drop_out)]
            else:
                x = [fc(x) for fc in self.fc]
        else:
            # encoder regularisation
            if self.training:
                x = self.drop_out(x)

            x = self.fc(x)
        return x