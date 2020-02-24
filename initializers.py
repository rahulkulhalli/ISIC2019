import torch.nn.init as init
import torch.nn as nn


def init_xavier_uniform(m):
    # Initialize weights to Glorot Uniform, bias to 0
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        # init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)
            
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        # init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)
            
    if isinstance(m, nn.Embedding):
        init.xavier_uniform_(m.weight.data)
        # init.xavier_normal_(m.weight.data)
        
def init_gamma(m):
    
    if isinstance(m, nn.Embedding):
        init.ones_(m.weight.data)
        
    if isinstance(m, nn.BatchNorm2d) and m.weight is not None:
        init.ones_(m.weight.data)

        
def init_beta(m):
    
    if isinstance(m, nn.Embedding):
        init.zeros_(m.weight.data)
        
    if isinstance(m, nn.BatchNorm2d) and m.bias is not None:
        init.zeros_(m.bias.data)
