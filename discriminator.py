import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
from blocks import *

'''
class Discriminator(nn.Module):
    
    def __init__(self, num_classes, use_spectral_norm=True, 
                 use_attention=False, base_filters=4, use_dropout=False):
        
        super(Discriminator, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_dropout = use_dropout
        
        self.block1 = res_block_d(3, base_filters, use_spectral_norm=use_spectral_norm) # 128x128
        self.block2 = res_block_d(base_filters, 2*base_filters, use_spectral_norm=use_spectral_norm) # 64x64
        self.block3 = res_block_d(2*base_filters, 4*base_filters, use_spectral_norm=use_spectral_norm) # 32x32
        self.block4 = res_block_d(4*base_filters, 8*base_filters, use_spectral_norm=use_spectral_norm) # 16x16
        self.block5 = res_block_d(8*base_filters, 16*base_filters, use_spectral_norm=use_spectral_norm) # 8x8
        
        final_linear = linear_layer(in_features=16*base_filters, out_features=1)
        self.final_linear = spectral_norm(final_linear) if use_spectral_norm else final_linear
        
        embedding = embedding_layer(num_embeddings=num_classes, embedding_dim=16*base_filters)
        self.embedding = spectral_norm(embedding) if use_spectral_norm else embedding
        
        if use_attention:
            self.att = SelfAttention(16*base_filters, downsample=True)
            self.att2 = SelfAttention(8*base_filters, downsample=True)
            
        if use_dropout:
            self.dpt1 = nn.Dropout2d(p=0.5)
            # self.dpt2 = nn.Dropout2d(p=0.35)
        
    def forward(self, x, y=None, visualize=False):
        
        x = self.block1(x)
        x = self.block2(x)
        
        # Note: Shifted SA from 2nd block op to 4th block op.
        x = self.block3(x)
        
        if self.use_dropout:
            x = self.dpt1(x)
        
        x = self.block4(x)
        
        if self.use_attention:
            if visualize:
                x, attn_map = self.att2(x, visualize=visualize)
            else:
                x = self.att2(x)
        
        x = self.block5(x)
            
        if self.use_attention:
            x = self.att(x)
        
        # x = self.block6(x)
        
        # Global Average Pooling
        x = x.sum(dim=(2, 3))
        
        output = self.final_linear(x)
        
        if y is not None:
            y_embedding= self.embedding(y)
            y_projection = (x * y_embedding).sum(dim=-1).unsqueeze(dim=-1)
            output += y_projection
            
        if visualize:
            return output, attn_map
        return output
'''

class Discriminator(nn.Module):
    
    def __init__(self, num_classes, use_spectral_norm=True, 
                 use_attention=False, base_filters=4, use_dropout=False):
        
        super(Discriminator, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_dropout = use_dropout
        
        self.block1 = res_block_d(3, base_filters, use_spectral_norm=use_spectral_norm) # 128x128
        self.block2 = res_block_d(base_filters, 2*base_filters, use_spectral_norm=use_spectral_norm) # 64x64
        self.block3 = res_block_d(2*base_filters, 4*base_filters, use_spectral_norm=use_spectral_norm) # 32x32
        self.block4 = res_block_d(4*base_filters, 8*base_filters, use_spectral_norm=use_spectral_norm) # 16x16
        self.block5 = res_block_d(8*base_filters, 16*base_filters, use_spectral_norm=use_spectral_norm) # 8x8
        
        final_linear = linear_layer(in_features=16*base_filters, out_features=1)
        self.final_linear = spectral_norm(final_linear) if use_spectral_norm else final_linear
        
        embedding = embedding_layer(num_embeddings=num_classes, embedding_dim=16*base_filters)
        self.embedding = spectral_norm(embedding) if use_spectral_norm else embedding
        
        if use_attention:
            # self.att = SelfAttention(16*base_filters, downsample=True)
            self.att2 = SelfAttention(2*base_filters, downsample=True)
            
        if use_dropout:
            self.dpt1 = nn.Dropout2d(p=0.5)
            # self.dpt2 = nn.Dropout2d(p=0.35)
        
    def forward(self, x, y=None, visualize=False):
        
        x = self.block1(x)
        x = self.block2(x)
        
        if self.use_attention:
            if visualize:
                x, attn_map = self.att2(x, visualize=visualize)
            else:
                x = self.att2(x)
        
        # Note: Shifted SA from 2nd block op to 4th block op.
        x = self.block3(x)
#         if self.use_attention:
#             x = self.att(x)
        
        if self.use_dropout:
            x = self.dpt1(x)
        
        x = self.block4(x)
        x = self.block5(x)
        # x = self.block6(x)
        
        # Global Average Pooling
        x = x.sum(dim=(2, 3))
        
        output = self.final_linear(x)
        
        if y is not None:
            y_embedding= self.embedding(y)
            y_projection = (x * y_embedding).sum(dim=-1).unsqueeze(dim=-1)
            output += y_projection
            
        if visualize:
            return output, attn_map
        return output


class Discriminator32(nn.Module):
    
    def __init__(self, num_classes, use_spectral_norm=True, 
                 use_attention=False, base_filters=64):
        
        super(Discriminator32, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        self.block1 = res_block_d(3, base_filters, use_spectral_norm=use_spectral_norm) # 16x16
        self.block2 = res_block_d(base_filters, base_filters, use_spectral_norm=use_spectral_norm) # 8x8
        self.block3 = res_block_d(base_filters, base_filters, use_spectral_norm=use_spectral_norm) # 4x4
        
        final_linear = linear_layer(in_features=base_filters, out_features=1)
        self.final_linear = spectral_norm(final_linear) if use_spectral_norm else final_linear
        
        embedding = embedding_layer(num_embeddings=num_classes, embedding_dim=base_filters)
        self.embedding = spectral_norm(embedding) if use_spectral_norm else embedding
        
        if use_attention:
            
            self.att = SelfAttention(base_filters)
        
    def forward(self, x, y=None):
        
        x = self.block1(x)
        x = self.block2(x)
        if self.use_attention:
            x = self.att(x)

        x = self.block3(x)
        
        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        
        # Global Average Pooling
        x = x.sum(dim=(2, 3))
        
        output = self.final_linear(x)
        
        if y is not None:
            y_embedding= self.embedding(y)
            y_projection = (x * y_embedding).sum(dim=-1).unsqueeze(dim=-1)
            output += y_projection
            
        return output
    
class Discriminator32_FakeClass(nn.Module):
    
    def __init__(self, num_classes, use_spectral_norm=True, 
                 use_attention=False, base_filters=64):
        
        super(Discriminator32_FakeClass, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        self.block1 = res_block_d(3, base_filters, use_spectral_norm=use_spectral_norm) # 16x16
        self.block2 = res_block_d(base_filters, base_filters, use_spectral_norm=use_spectral_norm) # 8x8
        self.block3 = res_block_d(base_filters, base_filters, use_spectral_norm=use_spectral_norm) # 4x4
        
        final_linear = linear_layer(in_features=base_filters, out_features=1)
        self.final_linear = spectral_norm(final_linear) if use_spectral_norm else final_linear
        
        embedding = embedding_layer(num_embeddings=num_classes, embedding_dim=base_filters)
        self.embedding = spectral_norm(embedding) if use_spectral_norm else embedding
        
        if use_attention:
            
            self.att = SelfAttention(base_filters)
        
    def forward(self, x, y=None, fake_y=None):
        
        x = self.block1(x)
        x = self.block2(x)
        if self.use_attention:
            x = self.att(x)

        x = self.block3(x)
        
        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        
        # Global Average Pooling
        x = x.sum(dim=(2, 3))
        
        output = self.final_linear(x)
        if fake_y is not None:
            output_copy = output
        
        if y is not None:
            y_embedding= self.embedding(y)
            y_projection = (x * y_embedding).sum(dim=-1).unsqueeze(dim=-1)
            output += y_projection
            
        if fake_y is not None:
            fake_y_embedding= self.embedding(fake_y)
            fake_y_projection = (x * fake_y_embedding).sum(dim=-1).unsqueeze(dim=-1)
            output_copy += fake_y_projection
            
        if fake_y is not None:
            return output, output_copy
            
        return output


    
class ConditionalDiscriminator(nn.Module):
    def __init__(self, NUM_CLASSES=10):
        super(ConditionalDiscriminator, self).__init__()
        self.embedding = nn.Embedding(NUM_CLASSES, 32*32*3)
        self.ndf = 64
        self.discriminator = self.build_discriminator()

        
    def build_discriminator(self):
        
        D = nn.Sequential(
            # 3 -> 8
            spectral_norm(nn.Conv2d(3, self.ndf*2, kernel_size=3, stride=2, 
                      padding=1, bias=False)),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 -> 16
            spectral_norm(nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=3, stride=2, 
                      padding=1, bias=False)),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 32
            spectral_norm(nn.Conv2d(self.ndf*4, self.ndf*8, kernel_size=3, stride=2, 
                      padding=1, bias=False)),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            SelfAttention(C=self.ndf*8),
            
            # 32 -> 64
            spectral_norm(nn.Conv2d(self.ndf*8, self.ndf*16, kernel_size=3, stride=2, 
                      padding=1, bias=False)),
            nn.BatchNorm2d(self.ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            
            SelfAttention(C=self.ndf*16),
            
            spectral_norm(nn.Conv2d(self.ndf*16, 1, kernel_size=2, stride=1, 
                      padding=0, bias=True))
        )
        
        return D
    
    
    def forward(self, x, class_labels):
        # class_label -> (B,)
        # x -> (B, 3, 224, 224)
        
        # embedded_labels -> (B, 224*224*3)
        embedded_labels = self.embedding(class_labels)
        
        # embedded_labels -> (B, 3, 224, 224)
        embedded_labels = embedded_labels.reshape((embedded_labels.size(0), 3, 32, 32))
        
        combined = embedded_labels * x
        
        outputs = self.discriminator(combined)
        
        return outputs.view(outputs.size(0), -1)

    
class ProjectionDiscriminator(nn.Module):
    def __init__(self, NUM_CLASSES=10):
        super(ProjectionDiscriminator, self).__init__()
        
        self.ndf = 64
        
        self.embedding = spectral_norm(nn.Embedding(NUM_CLASSES, self.ndf*16))
        
        self.discriminator = self.build_discriminator()
        self.final_linear = spectral_norm(nn.Linear(self.ndf*16, 1, bias=True))

        
    def build_discriminator(self):
        
        D = nn.Sequential(
            # 3 -> 8
            spectral_norm(nn.Conv2d(3, self.ndf*2, kernel_size=3, stride=2, 
                      padding=1, bias=False)),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 -> 16
            spectral_norm(nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=3, stride=2, 
                      padding=1, bias=False)),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 32
            spectral_norm(nn.Conv2d(self.ndf*4, self.ndf*8, kernel_size=3, stride=2, 
                      padding=1, bias=False)),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            SelfAttention(C=self.ndf*8),
            
            # 32 -> 64
            spectral_norm(nn.Conv2d(self.ndf*8, self.ndf*16, kernel_size=3, stride=2, 
                      padding=1, bias=False)),
            nn.BatchNorm2d(self.ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # SelfAttention(C=self.ndf*16),
        )
        
        return D
    
    
    def forward(self, x, class_labels=None):
        # class_label -> (B,)
        # x -> (B, 3, 224, 224)
        
        # embedded_labels -> (B, 224*224*3)
        embedded_labels = self.embedding(class_labels)
        
        # embedded_labels -> (B, 3, 224, 224)
        # embedded_labels = embedded_labels.reshape((embedded_labels.size(0), 3, 32, 32))
        
        # combined = embedded_labels * x
        
        x = self.discriminator(x)
        
        x = x.mean(dim=(2, 3))
        
        output = self.final_linear(x)
        
        if class_labels is not None:
            y_embedding= self.embedding(class_labels)
            y_projection = (x * y_embedding).sum(dim=-1).unsqueeze(dim=-1)
            output += y_projection
        
        
        return output.view(output.size(0), -1)
    
class ProjectionDiscriminator_nobn(nn.Module):
    def __init__(self, NUM_CLASSES=10):
        super(ProjectionDiscriminator_nobn, self).__init__()
        
        self.ndf = 64
        
        self.embedding = spectral_norm(nn.Embedding(NUM_CLASSES, self.ndf*16))
        
        self.discriminator = self.build_discriminator()
        self.final_linear = spectral_norm(nn.Linear(self.ndf*16, 1, bias=True))

        
    def build_discriminator(self):
        
        D = nn.Sequential(
            # 3 -> 8
            spectral_norm(nn.Conv2d(3, self.ndf*2, kernel_size=3, stride=2, 
                      padding=1, bias=False)),
            # nn.BatchNorm2d(self.ndf*2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            
            # 8 -> 16
            spectral_norm(nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=3, stride=2, 
                      padding=1, bias=False)),
            # nn.BatchNorm2d(self.ndf*4),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            
            # 16 -> 32
            spectral_norm(nn.Conv2d(self.ndf*4, self.ndf*8, kernel_size=3, stride=2, 
                      padding=1, bias=False)),
            # nn.BatchNorm2d(self.ndf*8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            
            SelfAttention(C=self.ndf*8),
            
            # 32 -> 64
            spectral_norm(nn.Conv2d(self.ndf*8, self.ndf*16, kernel_size=3, stride=2, 
                      padding=1, bias=False)),
            # nn.BatchNorm2d(self.ndf*16),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            
            # SelfAttention(C=self.ndf*16),
        )
        
        return D
    
    
    def forward(self, x, class_labels=None):
        # class_label -> (B,)
        # x -> (B, 3, 224, 224)
        
        # embedded_labels -> (B, 224*224*3)
        embedded_labels = self.embedding(class_labels)
        
        # combined = embedded_labels * x
        
        x = self.discriminator(x)
        
        x = x.mean(dim=(2, 3))
        
        output = self.final_linear(x)
        
        if class_labels is not None:
            y_embedding= self.embedding(class_labels)
            y_projection = (x * y_embedding).sum(dim=-1).unsqueeze(dim=-1)
            output += y_projection
        
        
        return output.view(output.size(0), -1)
