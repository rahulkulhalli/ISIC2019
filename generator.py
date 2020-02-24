import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm
from blocks import *

'''
class Generator(nn.Module):
    
    def __init__(self, num_classes, z_dim=64, base_width=8, base_filters=4, 
                 use_spectral_norm=True, use_attention=False):
        
        super(Generator, self).__init__()
        
        self.num_classes = num_classes
        self.base_width = base_width
        self.base_filters = base_filters
        
        self.use_attention = use_attention
        
        l1 = linear_layer(in_features=z_dim, out_features=(base_width ** 2) * 16 * base_filters)
        self.l1 = spectral_norm(l1) if use_spectral_norm else l1
        
        self.block0 = res_block_g(16*base_filters, 8*base_filters, num_classes, use_spectral_norm=use_spectral_norm) # 16x16
        self.block1 = res_block_g(8*base_filters, 4*base_filters, num_classes, use_spectral_norm=use_spectral_norm) # 32x32
        self.block2 = res_block_g(4*base_filters, 4*base_filters, num_classes, use_spectral_norm=use_spectral_norm) # 64x64
        self.block3 = res_block_g(4*base_filters, 2*base_filters, num_classes, use_spectral_norm=use_spectral_norm) # 128x128
        self.block4 = res_block_g(2*base_filters, base_filters, num_classes, use_spectral_norm=use_spectral_norm) # 256x256
        
        # self.bn = ConditionalBN(num_classes, base_filters)# batch_norm(4*base_filters)
        self.bn = batch_norm(base_filters)
        self.act = nn.ReLU()
        
        conv_l = conv_layer(base_filters, 3, 3, padding=1)
        self.conv_l = spectral_norm(conv_l) if use_spectral_norm else conv_l
        self.tanh = nn.Tanh()
        
        if use_attention:
            self.att = SelfAttention(4*base_filters, downsample=True)
            self.att2 = SelfAttention(4*base_filters, downsample=True)
        
        
    def forward(self, z, y=None, visualize=False):
        
        x = self.l1(z).view(-1, 16*self.base_filters, self.base_width, self.base_width)
        
        x = self.block0(x, y)
        x = self.block1(x, y)
        if self.use_attention:
            x = self.att(x)
        x = self.block2(x, y)
        if self.use_attention:
            if visualize:
                x, attn_map = self.att2(x, visualize=visualize)
            x = self.att2(x)
        x = self.block3(x, y)
        x = self.block4(x, y)
        
        x = self.act(self.bn(x))
        
        output = self.tanh(self.conv_l(x))
        
        if visualize:
            return output, attn_map
        return output
        
        
'''

class Generator(nn.Module):
    
    def __init__(self, num_classes, z_dim=64, base_width=8, base_filters=4, 
                 use_spectral_norm=True, use_attention=False):
        
        super(Generator, self).__init__()
        
        self.num_classes = num_classes
        self.base_width = base_width
        self.base_filters = base_filters
        
        self.use_attention = use_attention
        
        l1 = linear_layer(in_features=z_dim, out_features=(base_width ** 2) * 16 * base_filters)
        self.l1 = spectral_norm(l1) if use_spectral_norm else l1
        
        self.block0 = res_block_g(16*base_filters, 8*base_filters, num_classes, use_spectral_norm=use_spectral_norm) # 16x16
        self.block1 = res_block_g(8*base_filters, 4*base_filters, num_classes, use_spectral_norm=use_spectral_norm) # 32x32
        self.block2 = res_block_g(4*base_filters, 4*base_filters, num_classes, use_spectral_norm=use_spectral_norm) # 64x64
        self.block3 = res_block_g(4*base_filters, 2*base_filters, num_classes, use_spectral_norm=use_spectral_norm) # 128x128
        self.block4 = res_block_g(2*base_filters, base_filters, num_classes, use_spectral_norm=use_spectral_norm) # 256x256
        
        # self.bn = ConditionalBN(num_classes, base_filters)# batch_norm(4*base_filters)
        self.bn = batch_norm(base_filters)
        self.act = nn.ReLU()
        
        conv_l = conv_layer(base_filters, 3, 3, padding=1)
        self.conv_l = spectral_norm(conv_l) if use_spectral_norm else conv_l
        self.tanh = nn.Tanh()
        
        if use_attention:
            # self.att = SelfAttention(4*base_filters, downsample=True)
            self.att2 = SelfAttention(4*base_filters, downsample=True)
        
        
    def forward(self, z, y=None, visualize=False):
        
        x = self.l1(z).view(-1, 16*self.base_filters, self.base_width, self.base_width)
        
        x = self.block0(x, y)
        x = self.block1(x, y)
#         if self.use_attention:
#             x = self.att(x)
        x = self.block2(x, y)
        if self.use_attention:
            if visualize:
                x, attn_map = self.att2(x, visualize=visualize)
            x = self.att2(x)
        x = self.block3(x, y)
        x = self.block4(x, y)
        
        x = self.act(self.bn(x))
        
        output = self.tanh(self.conv_l(x))
        
        if visualize:
            return output, attn_map
        return output

class Generator32(nn.Module):
    
    def __init__(self, num_classes, z_dim=64, base_width=2, base_filters=64, 
                 use_spectral_norm=True, use_attention=False):
        
        super(Generator32, self).__init__()
        
        self.num_classes = num_classes
        self.base_width = base_width
        self.base_filters = base_filters
        
        self.use_attention = use_attention
        
        l1 = linear_layer(in_features=z_dim, out_features=(base_width ** 2) * base_filters)
        self.l1 = spectral_norm(l1) if use_spectral_norm else l1
        
        self.block0 = res_block_g(base_filters, base_filters, num_classes, use_spectral_norm=use_spectral_norm) # 8x8
        self.block1 = res_block_g(base_filters, base_filters, num_classes, use_spectral_norm=use_spectral_norm) # 16x16
        self.block2 = res_block_g(base_filters, base_filters, num_classes, use_spectral_norm=use_spectral_norm) #32x32
        # self.block3 = res_block_g(4*base_filters, 4*base_filters, num_classes, use_spectral_norm=use_spectral_norm) #32x32
        
        # self.bn = ConditionalBN(num_classes, base_filters)# batch_norm(4*base_filters)
        self.bn = batch_norm(base_filters)
        self.act = nn.ReLU()
        
        conv_l = conv_layer(base_filters, 3, 3, padding=1)
        self.conv_l = spectral_norm(conv_l) if use_spectral_norm else conv_l
        self.tanh = nn.Tanh()
        
        if use_attention:
            
            self.att = SelfAttention(base_filters)
        
        
    def forward(self, z, y=None):
        
        x = self.l1(z).view(-1, self.base_filters, self.base_width, self.base_width)
        
        x = self.block0(x, y)
        x = self.block1(x, y)
        if self.use_attention:
            x = self.att(x)
        x = self.block2(x, y)
        # x = self.block3(x, y)
        
        # x = self.block4(x, y)
        # x = self.block5(x, y)
        
        # x = self.act(self.bn(x, y))
        x = self.act(self.bn(x))
        
        output = self.tanh(self.conv_l(x))
        return output

class ConditionalGenerator(nn.Module):
    def __init__(self, nZ=10, NUM_CLASSES=10):
        super(ConditionalGenerator, self).__init__()
        self.nZ = nZ
        self.ndf = 64
        self.embedding = spectral_norm(nn.Embedding(num_embeddings=NUM_CLASSES, embedding_dim=self.nZ))
        self.generator = self.build_generator()
    
    def build_generator(self):
        # merged_x -> (B, nZ, 1, 1)
        
        nZ = self.nZ
        
        G = nn.Sequential(
            # IN: (-1, nZ, 1, 1)
            spectral_norm(nn.ConvTranspose2d(nZ, self.ndf*32, kernel_size=4, stride=2, 
                               bias=False, padding=0)),
            nn.BatchNorm2d(self.ndf*32),
            nn.ReLU(inplace=True),
            
            spectral_norm(nn.ConvTranspose2d(self.ndf*32, self.ndf*16, kernel_size=4, stride=2, 
                               bias=False, padding=1)),
            nn.BatchNorm2d(self.ndf*16),
            nn.ReLU(inplace=True),
            
            SelfAttention(C=self.ndf*16),
            
            spectral_norm(nn.ConvTranspose2d(self.ndf*16, self.ndf*8, kernel_size=4, stride=2, 
                               padding=1, bias=False)),
            nn.BatchNorm2d(self.ndf*8),
            nn.ReLU(inplace=True),
            
            SelfAttention(C=self.ndf*8),
            
            spectral_norm(nn.ConvTranspose2d(self.ndf*8, self.ndf*4, kernel_size=4, stride=2, 
                               padding=1, bias=False)),
            nn.BatchNorm2d(self.ndf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, 
                      padding=0, bias=True),
            nn.Tanh()
        )
        
        return G
        
    
    def forward(self, x, label_in):
        # x -> (B, nZ, 1, 1)
        # label -> (B,)
        
        # label_embedded -> (B, nZ)
        label_embedded = self.embedding(label_in)
        
        # label_embedded -> (B, nZ, 1, 1)
        label_embedded = label_embedded.unsqueeze(-1).unsqueeze(-1)
        
        assert label_embedded.size() == x.size()
        
        # Element-wise multiply.
        merged_in = label_embedded * x
        
        return self.generator(merged_in)


class ConditionalBNGenerator(nn.Module):
    def __init__(self, nZ=10, NUM_CLASSES=10):
        super(ConditionalBNGenerator, self).__init__()
        self.nZ = nZ
        self.ndf = 64
        self.num_classes = NUM_CLASSES
        # self.embedding = spectral_norm(nn.Embedding(num_embeddings=NUM_CLASSES, embedding_dim=self.nZ))
        self.generator = self.build_generator()
    
    def build_generator(self):
        # merged_x -> (B, nZ, 1, 1)
        
        nZ = self.nZ
        
        self.conv1 = spectral_norm(nn.ConvTranspose2d(nZ, self.ndf*32, kernel_size=4, stride=2, 
                               bias=False, padding=0))
        self.bn1 = ConditionalBN(self.num_classes, self.ndf*32)
        self.act1 = nn.ReLU()
        
        self.conv2 = spectral_norm(nn.ConvTranspose2d(self.ndf*32, self.ndf*16, kernel_size=4, stride=2, 
                               bias=False, padding=1))
        self.bn2 = ConditionalBN(self.num_classes, self.ndf*16)
        self.act2 = nn.ReLU()
        
        self.attn1 = SelfAttention(C=self.ndf*16)
        
        self.conv3 = spectral_norm(nn.ConvTranspose2d(self.ndf*16, self.ndf*8, kernel_size=4, stride=2, 
                               padding=1, bias=False))
        self.bn3 = ConditionalBN(self.num_classes, self.ndf*8)
        self.act3 = nn.ReLU()
        
        self.attn2 = SelfAttention(C=self.ndf*8)
        
        self.conv4 = spectral_norm(nn.ConvTranspose2d(self.ndf*8, self.ndf*4, kernel_size=4, stride=2, 
                               padding=1, bias=False))
        self.bn4 = ConditionalBN(self.num_classes, self.ndf*4)
        self.act4 = nn.ReLU()
        
        self.conv5 = nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, 
                      padding=0, bias=True)
        self.tanh = nn.Tanh()
        
    
    def forward(self, x, label_in):
        
        x = self.act1(self.bn1(self.conv1(x), label_in))
        x = self.act2(self.bn2(self.conv2(x), label_in))
        x = self.attn1(x)
        x = self.act3(self.bn3(self.conv3(x), label_in))
        x = self.attn2(x)
        x = self.act4(self.bn4(self.conv4(x), label_in))
        
        x = self.conv5(x)
        
        return self.tanh(x)
