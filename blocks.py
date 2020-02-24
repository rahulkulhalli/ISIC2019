import torch
import torch.nn as nn
import initializers as init
from torch.nn.utils.spectral_norm import spectral_norm


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias=True,
               padding_mode='zeros',):
    
    '''Utility function to initialize conv layer with Xavier Unifirm initializer'''
    
    conv = nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels, 
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias,
                     padding_mode=padding_mode)
    conv.apply(init.init_xavier_uniform)
    
    return conv


def linear_layer(in_features, out_features, bias=True):
    
    '''Utility function for initializing linear layer with Xavier Uniform initializer'''
    
    linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
    linear.apply(init.init_xavier_uniform)
    
    return linear


def embedding_layer(num_embeddings,
                    embedding_dim,
                    padding_idx=None,
                    max_norm=None,
                    norm_type=2.0,
                    scale_grad_by_freq=False,
                    sparse=False,
                    _weight=None,):
    
    '''Utility function for initializing embedding layer with Xavier Uniform initializer'''
    embedding = nn.Embedding(num_embeddings=num_embeddings,
                             embedding_dim=embedding_dim,
                             padding_idx=padding_idx,
                             max_norm=max_norm,
                             norm_type=norm_type,
                             scale_grad_by_freq=scale_grad_by_freq,
                             sparse=sparse,
                             _weight=_weight)
    
    embedding.apply(init.init_xavier_uniform)
    
    return embedding


def batch_norm(num_features,
               eps=1e-05,
               momentum=0.1,
               affine=True,
               track_running_stats=True,):
    
    '''Utility function to initialize batch norm layer with ones and zeros'''
    bn = nn.BatchNorm2d(num_features,
                        eps=eps,
                        momentum=momentum,
                        affine=affine,
                        track_running_stats=track_running_stats,)
    bn.apply(init.init_gamma)
    bn.apply(init.init_beta)
    
    return bn

"""
class SelfAttention(nn.Module):
    
    def __init__(self, C, use_spectral_norm=True):
        super(SelfAttention, self).__init__()
        self.C = C
        self.gamma = nn.Parameter(torch.zeros(1))
        q_conv = conv_layer(self.C, self.C // 8, kernel_size=1)
        self.q_conv = spectral_norm(q_conv) if use_spectral_norm else q_conv
        
        k_conv = conv_layer(self.C, self.C // 8, kernel_size=1)
        self.k_conv = spectral_norm(k_conv) if use_spectral_norm else k_conv
        
        v_conv = conv_layer(self.C, self.C, kernel_size=1)
        self.v_conv = spectral_norm(v_conv) if use_spectral_norm else v_conv
        
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, x):
        
        # Note: H*W == N
        
        # BxCxWxH
        batch_size, C, H, W = x.size()
        
        '''
        ---------
        conv_f_x
        ---------
        In: BxCxWxH
        conv_Out: Bx(C//8)xWxH
        View_Out: Bx(C//8)x(W*H)
        Permute_Out: BxNx(C//8)
        '''
        conv_f_x = self.q_conv(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        
        '''
        ---------
        conv_g_x
        ---------
        In: BxCxWxH
        conv_Out: Bx(C//8)xWxH
        View_Out: Bx(C//8)xN
        '''
        conv_g_x = self.k_conv(x).view(batch_size, -1, H*W)
        
        '''
        logits_Out: BxNxN
        '''
        logits = torch.bmm(conv_f_x, conv_g_x)
        
        '''
        attn_map_Out: BxNxN
        '''
        attn_map = self.softmax(logits)
        
        '''
        ---------
        conv_h_x
        ---------
        In: BxCxWxH
        conv_Out: BxCxWxH
        View_Out: BxCxN
        '''
        conv_h_x = self.v_conv(x).view(batch_size, -1, H*W)
        
        '''
        attn_map_Permute: BxNxN
        out_Out: BxCxN
        
        (CxN)x(NxN) => (CxN)
        '''
        out = torch.bmm(conv_h_x, attn_map.permute(0, 2, 1))
        
        '''
        out_Out: BxCxWxH
        '''
        out = out.view(batch_size, C, W, H)
        
        return (self.gamma * out) + x

"""
    
#########################
# Paper implementation
#########################

class SelfAttention(nn.Module):
    
    def __init__(self, C, use_spectral_norm=True, downsample=True):
        super(SelfAttention, self).__init__()
        self.C = C
        
        self.downsample = downsample
        
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        theta_conv = conv_layer(self.C, self.C // 8, kernel_size=1)
        self.theta_conv = spectral_norm(theta_conv) if use_spectral_norm else theta_conv
        
        phi_conv = conv_layer(self.C, self.C // 8, kernel_size=1)
        
        if self.downsample:
            self.mPool1 = nn.MaxPool2d(2)
            self.mPool2 = nn.MaxPool2d(2)
        
        self.phi_conv = spectral_norm(phi_conv) if use_spectral_norm else phi_conv
        
        g_conv = conv_layer(self.C, self.C // 2, kernel_size=1)
        self.g_conv = spectral_norm(g_conv) if use_spectral_norm else g_conv
        
        self.softmax = nn.Softmax(dim=-1)
        
        final_conv = conv_layer(self.C // 2, self.C, kernel_size=1)
        self.final_conv = spectral_norm(final_conv) if use_spectral_norm else final_conv
        
        
    def forward(self, x, visualize=False):
        
        # Note: H*W == N
        
        # BxCxWxH
        batch_size, C, H, W = x.size()
     
        '''
        ---------
        conv_g_x
        ---------
        In: BxCxWxH
        Conv_Out: Bx(C//8)xWxH
        View_Out: Bx(C//8)x(WxH)
        Permute_Out: Bx(WxH)x(C//8)
        '''
        conv_theta = self.theta_conv(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        
        '''
        ---------
        conv_f_x
        ---------
        In: BxCxWxH
        conv_Out: Bx(C//8)xWxH
        mPool_Out: Bx(C//8)xW//2xH//2
        View_Out: Bx(C//8)x(W//2xH//2)
        Permute_out: Bx(W//2xH//2)x(C//8)
        '''
        conv_phi = self.phi_conv(x)
        
        if self.downsample:
            conv_phi = self.mPool1(conv_phi)
        
        c_out = conv_phi.size()[1]
        
        conv_phi = conv_phi.view(batch_size, c_out, -1)
        
        '''
        logits_Out: BxNxN//2
        '''
        logits = torch.bmm(conv_theta, conv_phi)
        
        '''
        attn_map_Out: BxNxN//2
        '''
        attn_map = self.softmax(logits)
        
        '''
        ---------
        conv_h_x
        ---------
        In: BxCxWxH
        conv_Out: BxC//2xWxH
        MPool_Out: BxC//2x(H//2xW//2)
        Transpose_Out: Bx(H//2xW//2)xC//2
        
        '''
        g = self.g_conv(x)
        
        c_out = g.size()[1]
        
        if self.downsample:
            g = self.mPool2(g)
        
        g = g.view(batch_size, c_out, -1).permute(0, 2, 1)
        
        
        '''
        Attn: Bx(HxW)x(H//2xW//2)
        G: Bx(H//2xW//2)xC//2
        
        Out => Bx(HxW)xC//2
        Out_permute -> BxC//2x(HxW)
        '''
        out = torch.bmm(attn_map, g).permute(0, 2, 1).view(batch_size, -1, H, W)
        
        '''
        final_conv_out: BxCxHxW
        '''
        final_conv_out = self.final_conv(out)
        
        if visualize:
            return (self.gamma * final_conv_out) + x, attn_map
        return (self.gamma * final_conv_out) + x



'''
class SelfAttention(nn.Module):
    
    """ Self attention Layer"""
    def __init__(self, C, use_spectral_norm=False):
        super(SelfAttention,self).__init__()
       
        self.query_conv = nn.Conv2d(in_channels = C , out_channels = C//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = C , out_channels = C//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = C , out_channels = C , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out
'''  
    
class ConditionalBN(nn.Module):
    
    def __init__(self, num_classes, num_features):
        
        # Here, we purposely set affine=False since we wont be needing the weights and
        # biases of the standard BatchNorm
        
        super(ConditionalBN, self).__init__()
        
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        
        self.gammas = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_features)
        self.gammas.apply(init.init_gamma)
        self.betas = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_features)
        self.betas.apply(init.init_beta)
        
    def forward(self, x, y):
        
        x = self.bn(x)
        
        # Gamma
        weight = self.gammas(y).unsqueeze(dim=-1).unsqueeze(dim=-1)
        # Beta
        bias = self.betas(y).unsqueeze(dim=-1).unsqueeze(dim=-1)
        
        return weight * x + bias
    
    
# This code is mostly derived from https://github.com/pfnet-research/sngan_projection/blob/master/gen_models/resblocks.py
# and https://github.com/pfnet-research/sngan_projection/blob/master/dis_models/resblocks.py, which is written in Chainer

class res_block_g(nn.Module):
    
    def __init__(self, in_channels, out_channels, n_classes, kernel=3,
                 use_spectral_norm=True, upsample=True,
                 negative_slope=0.1):
        
        super(res_block_g, self).__init__()
        
        # self.bn0 = nn.BatchNorm2d(in_channels) # ConditionalBN(n_classes, in_channels)
        self.bn0 = ConditionalBN(n_classes, in_channels)
        self.relu0 = nn.ReLU()
        
        conv1 = conv_layer(in_channels, out_channels, kernel, padding=1)
        self.conv1 = spectral_norm(conv1) if use_spectral_norm else conv1
        # self.bn1 = nn.BatchNorm2d(out_channels) # ConditionalBN(n_classes, num_features=out_channels)
        self.bn1 = ConditionalBN(n_classes, num_features=out_channels)
        self.act1 = nn.ReLU()
        
        conv2 = conv_layer(out_channels, out_channels, kernel, padding=1)
        self.conv2 = spectral_norm(conv2) if use_spectral_norm else conv2
        
        self.learnable_sc = (in_channels != out_channels) or upsample
        if self.learnable_sc:
            conv_sc = conv_layer(in_channels, out_channels, 1, padding=0)
            self.conv_sc = spectral_norm(conv_sc) if use_spectral_norm else conv_sc
        
        if upsample:
            self.upsample0 = nn.Upsample(scale_factor=2)
            self.upsample1 = nn.Upsample(scale_factor=2)
            
        self.upsample = upsample
            
    def residual(self, x, y):
        
        # Check here lol
        x = self.relu0(self.bn0(x, y))
        # x = self.relu0(self.bn0(x))
        
        if self.upsample:
            x = self.upsample0(x)
        
        # Check here lol
        x = self.act1(self.bn1(self.conv1(x), y))
        # x = self.act1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        
        return x
    
    def shortcut(self, x):
        
        if self.upsample:
            x = self.upsample1(x)
        return self.conv_sc(x) if self.learnable_sc else x
        
    def forward(self, x, y):
        
        return self.residual(x, y) + self.shortcut(x)

# --------------------
# This is the residual block with Conv2DTranspose instead of Upsample+Conv
# --------------------
    
class res_block_g_C2DT(nn.Module):
    
    # block1 -> Conv2DT + CBN + ReLU -> Conv-----------
        #        |                                    |
        #        |                                    + --- CBN + ReLU -> block2
        #        |                                    |
        #       Conv2DT ------------------------------
    
    def __init__(self, in_channels, out_channels, n_classes, kernel=3,
                 use_spectral_norm=True, upsample=True):
        
        super(res_block_g_C2DT, self).__init__()
        
        self.bn0 = ConditionalBN(n_classes, in_channels)
        self.relu0 = nn.ReLU()
        
        # Double spacial dimensions
        self.c2dt_residual = nn.ConvTranspose2d(in_channels, out_channels, 
                                       kernel_size=3, padding=1, output_padding=1, 
                                       stride=2)
        
        self.c2dt_residual = spectral_norm(self.c2dt_residual) if use_spectral_norm else self.c2dt_residual
        
        # Double spacial dimensions
        self.c2dt_shortcut = nn.ConvTranspose2d(in_channels, out_channels, 
                                       kernel_size=3, padding=1, output_padding=1, 
                                       stride=2)
        
        self.c2dt_shortcut = spectral_norm(self.c2dt_shortcut) if use_spectral_norm else self.c2dt_shortcut
        
        conv = conv_layer(out_channels, out_channels, kernel, padding=1, stride=1)
        self.conv = spectral_norm(conv) if use_spectral_norm else conv
        
        self.bn1 = ConditionalBN(n_classes, num_features=out_channels)
        self.act1 = nn.ReLU()
        
        '''
        self.learnable_sc = (in_channels != out_channels) or upsample
        if self.learnable_sc:
            conv_sc = conv_layer(in_channels, out_channels, 1, padding=0)
            self.conv_sc = spectral_norm(conv_sc) if use_spectral_norm else conv_sc
        
        if upsample:
            self.upsample0 = nn.Upsample(scale_factor=2)
            self.upsample1 = nn.Upsample(scale_factor=2)
            
        self.upsample = upsample
        '''
            
    def residual(self, x, y):
        
        _x = self.relu0(self.bn0(x, y))
        
        _x = self.act1(self.bn1(self.c2dt_residual(_x), y))
        
        _x = self.conv(_x)
        
        '''
        x = self.act1(self.bn1(self.conv1(x), y))
        x = self.conv2(x)
        '''
        
        return _x
    
    
    def shortcut(self, x):
        
        '''
        if self.upsample:
            x = self.upsample1(x)
        return self.conv_sc(x) if self.learnable_sc else x
        '''
        
        return self.c2dt_shortcut(x)
    
        
    def forward(self, x, y):
        
        return self.residual(x, y) + self.shortcut(x)
    
    
class OptimizedG_Block(nn.Module):
    
    # block1 -> Conv2DT + CBN + ReLU -> Conv-----------
        #        |                                    |
        #        |                                    + --- CBN + ReLU -> block2
        #        |                                    |
        #       Conv2DT ------------------------------
    
    def __init__(self, in_channels, out_channels, n_classes, kernel=3,
                 use_spectral_norm=True, upsample=True):
        
        super(OptimizedG_Block, self).__init__()
        
        self.bn0 = ConditionalBN(n_classes, in_channels)
        self.relu0 = nn.ReLU()
        
        # Double spacial dimensions
        self.c2dt_residual = nn.ConvTranspose2d(in_channels, out_channels, 
                                       kernel_size=3, padding=1, output_padding=1, 
                                       stride=2)
        
        self.c2dt_residual = spectral_norm(self.c2dt_residual) if use_spectral_norm else self.c2dt_residual
        
        # Double spacial dimensions
        self.c2dt_shortcut = nn.ConvTranspose2d(in_channels, out_channels, 
                                       kernel_size=3, padding=1, output_padding=1, 
                                       stride=2)
        
        self.c2dt_shortcut = spectral_norm(self.c2dt_shortcut) if use_spectral_norm else self.c2dt_shortcut
        
        conv = conv_layer(in_channels, out_channels, kernel, padding=1, stride=1)
        self.conv = spectral_norm(conv) if use_spectral_norm else conv
        
        self.bn1 = ConditionalBN(n_classes, num_features=out_channels)
        self.act1 = nn.ReLU()
        
        '''
        self.learnable_sc = (in_channels != out_channels) or upsample
        if self.learnable_sc:
            conv_sc = conv_layer(in_channels, out_channels, 1, padding=0)
            self.conv_sc = spectral_norm(conv_sc) if use_spectral_norm else conv_sc
        
        if upsample:
            self.upsample0 = nn.Upsample(scale_factor=2)
            self.upsample1 = nn.Upsample(scale_factor=2)
            
        self.upsample = upsample
        '''
            
    def residual(self, x, y):
        
        _x = self.act1(self.bn1(self.c2dt_residual(x), y))
        
        _x = self.conv(_x)
        
        '''
        x = self.act1(self.bn1(self.conv1(x), y))
        x = self.conv2(x)
        '''
        
        return _x
    
    
    def shortcut(self, x):
        
        '''
        if self.upsample:
            x = self.upsample1(x)
        return self.conv_sc(x) if self.learnable_sc else x
        '''
        
        return self.c2dt_shortcut(x)
    
        
    def forward(self, x, y):
        
        return self.residual(x, y) + self.shortcut(x)

# -------------------   

class res_block_d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel=3,
                 use_spectral_norm=True, downsample=True,
                 negative_slope=0.1):
        
        super(res_block_d, self).__init__()
        
        self.relu0 = nn.ReLU()
        
        conv1 = conv_layer(in_channels, out_channels, kernel, padding=1)
        self.conv1 = spectral_norm(conv1) if use_spectral_norm else conv1
        self.act1 = nn.ReLU()
        
        conv2 = conv_layer(out_channels, out_channels, 1, padding=0)
        self.conv2 = spectral_norm(conv2) if use_spectral_norm else conv2
        
        self.learnable_sc = (in_channels != out_channels) or downsample
        if self.learnable_sc:
            conv_sc = conv_layer(in_channels, out_channels, kernel, padding=1)
            self.conv_sc = spectral_norm(conv_sc) if use_spectral_norm else conv_sc
        
        if downsample:
            self.downsample0 = nn.AvgPool2d(kernel_size=2)
            self.downsample1 = nn.AvgPool2d(kernel_size=2)
            
        self.downsample = downsample
            
    def residual(self, x):
        
        x = self.relu0(x)
        
        if self.downsample:
            x = self.downsample0(x)
        
        x = self.act1(self.conv1(x))
        x = self.conv2(x)
        
        return x
    
    def shortcut(self, x):
        
        if self.downsample:
            x = self.downsample1(x)
        return self.conv_sc(x) if self.learnable_sc else x
        
    def forward(self, x):
        
        return self.residual(x) + self.shortcut(x)
    
# -----------------------

class res_block_d_stridedConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel=3,
                 use_spectral_norm=True):
        
        super(res_block_d_stridedConv, self).__init__()
        
        self.relu0 = nn.ReLU()
        
        conv1 = conv_layer(in_channels, out_channels, kernel, padding=1, stride=2)
        self.conv1 = spectral_norm(conv1) if use_spectral_norm else conv1
        self.act1 = nn.ReLU()
        
        # same i/p -> o/p dimensions
        conv = conv_layer(out_channels, out_channels, kernel, padding=1, stride=1)
        self.conv = spectral_norm(conv) if use_spectral_norm else conv
        
        conv2 = conv_layer(in_channels, out_channels, kernel, padding=1, stride=2)
        self.conv2 = spectral_norm(conv2) if use_spectral_norm else conv2
        
        '''
        self.learnable_sc = (in_channels != out_channels) or downsample
        if self.learnable_sc:
            conv_sc = conv_layer(in_channels, out_channels, kernel, padding=1)
            self.conv_sc = spectral_norm(conv_sc) if use_spectral_norm else conv_sc
        
        if downsample:
            self.downsample0 = nn.AvgPool2d(kernel_size=2)
            self.downsample1 = nn.AvgPool2d(kernel_size=2)
            
        self.downsample = downsample
        '''
            
    def residual(self, x):
        
        _x = self.relu0(x)
        
        '''
        if self.downsample:
            _x = self.downsample0(_x)
        '''
        
        _x = self.act1(self.conv1(_x))
        _x = self.conv(_x)
        
        return _x
    
    def shortcut(self, x):
        '''
        if self.downsample:
            x = self.downsample1(x)
        return self.conv_sc(x) if self.learnable_sc else x
        '''
        
        return self.conv2(x)
        
    def forward(self, x):
        
        return self.residual(x) + self.shortcut(x)
    

class OptimizedD_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel=3,
                 use_spectral_norm=True):
        
        super(OptimizedD_Block, self).__init__()
        
        self.relu0 = nn.ReLU()
        
        conv1 = conv_layer(in_channels, out_channels, kernel, padding=1, stride=2)
        self.conv1 = spectral_norm(conv1) if use_spectral_norm else conv1
        self.act1 = nn.ReLU()
        
        # same i/p -> o/p dimensions
        conv = conv_layer(out_channels, out_channels, kernel, padding=1, stride=1)
        self.conv = spectral_norm(conv) if use_spectral_norm else conv
        
        conv2 = conv_layer(in_channels, out_channels, kernel, padding=1, stride=2)
        self.conv2 = spectral_norm(conv2) if use_spectral_norm else conv2
        
        '''
        self.learnable_sc = (in_channels != out_channels) or downsample
        if self.learnable_sc:
            conv_sc = conv_layer(in_channels, out_channels, kernel, padding=1)
            self.conv_sc = spectral_norm(conv_sc) if use_spectral_norm else conv_sc
        
        if downsample:
            self.downsample0 = nn.AvgPool2d(kernel_size=2)
            self.downsample1 = nn.AvgPool2d(kernel_size=2)
            
        self.downsample = downsample
        '''
            
    def residual(self, x):
        
        '''
        if self.downsample:
            _x = self.downsample0(_x)
        '''
        
        _x = self.act1(self.conv1(x))
        _x = self.conv(_x)
        
        return _x
    
    def shortcut(self, x):
        '''
        if self.downsample:
            x = self.downsample1(x)
        return self.conv_sc(x) if self.learnable_sc else x
        '''
        
        return self.conv2(x)
        
    def forward(self, x):
        
        return self.residual(x) + self.shortcut(x)