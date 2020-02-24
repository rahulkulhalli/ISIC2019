class Config():
    
    def __init__(self):
        
        # Declare parameters
        self.z_dim = 64
        self.num_classes = 8
        self.base_width = 8
        self.base_filters = 8
        self.use_spectral_norm = True
        self.use_attention = True
        self.use_dropout = False
        
        self.g_lr = 0.0001
        # self.g_lr = 2.8242953648100018e-05
        
        self.d_lr = 0.0004
        # self.d_lr = 0.00011297181459240007
        
        # 1500 * 170 = 255k
        # 1mn/170 = 5.9k epochs lol
        self.epochs = 1500
        self.start_epoch = 0 # 509
        
        # No decay
        self.decay = 1.0
        
        self.beta1 = 0.1
        self.beta2 = 0.9
        self.d_updates_per_g = 2
        
        self.batch_size = 128
        
        # Dataset
        self.train_root = "/home/antpc/Documents/data_mount/ISIC2019/ISIC2019/Data_SAGAN/train"
        self.test_root = "/home/antpc/Documents/data_mount/ISIC2019/ISIC2019/Data_SAGAN/test"
        
        self.sample_every_x_epochs = 1
        self.sample_every_x_iters = 50
        self.sample_im_size = 256
        self.decay_every_x_epochs = 20
        self.show_every = 20
        self.sample_path = '.'
        self.log_path = '.'
        self.checkpoint_path = 'Dumps/V3/'
        self.checkpoint_every = 1
        
        self.device = 'cuda:0'
        self.data_parallel = True
        
        self.pretrained = True # if self.start_epoch > 0 else False