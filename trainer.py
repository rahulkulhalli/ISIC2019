import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

from utils import make_grid


class Trainer():
    
    def __init__(self, config, data_loader, samples_per_class=5, 
                 device='cuda:0', data_parallel=True, generator=None, discriminator=None):
        
        self.data_loader = data_loader
        self.device = device
        self.data_parallel = data_parallel
        self.samples_per_class = samples_per_class
        
        # Model parameters
        self.z_dim = config.z_dim
        self.num_classes = config.num_classes
        self.base_width = config.base_width
        self.base_filters = config.base_filters
        self.use_spectral_norm = config.use_spectral_norm
        self.use_attention = config.use_attention
        self.use_dropout = config.use_dropout
        
        # Training parameters
        self.batch_size = config.batch_size
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.epochs = config.epochs
        
        # Adam optimizer parameters
        self.decay = config.decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.decay = config.decay
        self.d_updates_per_g = config.d_updates_per_g
        
        # Sanity variables.
        self.current_epoch = config.start_epoch
        self.start_epoch = config.start_epoch
        
        # Logging
        # Keep a list of generator and discriminator loss per iteration
        self.d_loss = []
        self.d_loss_fake = []
        self.d_loss_real = []
        self.g_loss = []
        
        self.decay_every_x_epochs = config.decay_every_x_epochs
        self.sample_every_x_epochs = config.sample_every_x_epochs
        self.sample_every_x_iters = config.sample_every_x_iters
        
        self.sample_path = config.sample_path
        self.sample_im_size = config.sample_im_size
        self.log_path = config.log_path
        self.checkpoint_path = config.checkpoint_path
        
        if generator is None:
            assert discriminator is None
        if discriminator is None:
            assert generator is None
            
        self.generator = generator
        self.discriminator = discriminator
        
        # Set-up the optimizers
        self.setup_training()
        
        self.checkpoint_every = config.checkpoint_every
        self.show_every = config.show_every
        self.samples = []
    
    
    def checkpoint_models(self):
        torch.save(self.generator.module.state_dict(), 
                   self.checkpoint_path + 'models/generator_{}.pth'.format(self.current_epoch))
        torch.save(self.discriminator.module.state_dict(), 
                   self.checkpoint_path + 'models/discriminator_{}.pth'.format(self.current_epoch))
        
    def setup_training(self):
        
        self.g_opt = optim.Adam(self.generator.parameters(), lr=self.g_lr, 
                                betas=(self.beta1, self.beta2))
        self.d_opt = optim.Adam(self.discriminator.parameters(), lr=self.d_lr, 
                                betas=(self.beta1, self.beta2))
        
    def train(self):
        
        for epoch in range(self.start_epoch, self.epochs):
            
            # Update current epoch counter.
            self.current_epoch = epoch
            
            for iteration, (X, y) in enumerate(self.data_loader):
                
                batch_size = int(X.size()[0])
                
                # Real
                X = torch.FloatTensor(X).to(self.device)
                y = torch.LongTensor(y).to(self.device)
                
                for i in range(self.d_updates_per_g):

                    # --------Train the Discriminator ---------------------
                    
                    # Generate a fake batch
                    z = torch.FloatTensor(size=(batch_size, self.z_dim)).normal_(0., 1.).to(self.device)
                    
                    fake_X = self.generator(z, y).detach()

                    real_score = self.discriminator(X, y)
                    d_loss_real = (torch.nn.ReLU()(1. - real_score)).mean()
                    self.d_loss_real.append(d_loss_real.item())

                    fake_score_d = self.discriminator(fake_X, y)
                    d_loss_fake = (torch.nn.ReLU()(1. + fake_score_d)).mean()
                    self.d_loss_fake.append(d_loss_fake.item())

                    d_loss = d_loss_real + d_loss_fake
                    self.d_loss.append(d_loss.item())

                    # Clear gradient
                    self.d_opt.zero_grad()
                    self.g_opt.zero_grad()
                    d_loss.backward()
                    self.d_opt.step()
                    
                # --------Train the Generator ---------------------

                # sample another z
                z = torch.FloatTensor(size=(batch_size, self.z_dim)).normal_(0., 1.).to(self.device)

                fake_X = self.generator(z, y)

                fake_score_g = self.discriminator(fake_X, y)

                g_loss = -fake_score_g.mean()
                self.g_loss.append(g_loss.item())

                self.g_opt.zero_grad()
                self.d_opt.zero_grad()
                g_loss.backward()
                self.g_opt.step()
                
                if (iteration % 50) == 0:
                    print('Epoch: {} | Iteration: {} | D Loss: {} [D(x): {} | D(G(z)): {}] | G Loss: {}'.format(
                        epoch, iteration, d_loss.item(), real_score.mean().item(), fake_score_d.mean().item(), g_loss.item()
                    ))
                
            # --------Sample ---------------------
            if ((epoch % self.sample_every_x_epochs) == 0):
                self.sample()
                
            # Checkpoint
            if ((epoch % self.checkpoint_every) == 0):
                self.checkpoint_models()
                
            # Decay LR
            if epoch > 0 and epoch % self.decay_every_x_epochs == 0:
                print("Decaying Learning Rates...")
                
                self.g_opt.param_groups[0]['lr'] *= self.decay
                self.d_opt.param_groups[0]['lr'] *= self.decay
                
                print("New D_optim LR: {} | New G_optim LR: {}".format(
                    self.d_opt.param_groups[0]['lr'],
                    self.g_opt.param_groups[0]['lr']
                ))
                            
            print(50*'=')
    
                            
    def sample(self, padding=2):
        
        dpi = 100
        
        try:
            fake_X = self.generator(self.fixed_z, self.sample_y).detach().cpu()
            
            images = make_grid(fake_X, samples_per_class=self.samples_per_class, 
                               num_classes=self.num_classes, padding=padding, 
                               im_size=self.sample_im_size)
            
            if (self.current_epoch % self.checkpoint_every) == 0:
                
                y_labels = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
                y_ptr = self.sample_im_size//2
                y_locs = []
                
                while y_ptr < images.shape[0]:
                    y_locs.append(y_ptr)
                    y_ptr += (self.sample_im_size + padding)
                
                assert len(y_labels) == len(y_locs)
                
                fig, ax = plt.subplots(figsize=(images.shape[1]/dpi, images.shape[0]/dpi), dpi=dpi)
                ax.imshow(images, interpolation='nearest')
                ax.set_xticks([])
                plt.yticks(ticks=y_locs, labels=y_labels)
                plt.savefig(self.checkpoint_path + 'samples/{}.png'.format(self.current_epoch), bbox_inches='tight', dpi=dpi)
                
                # CLOSE THE FIGURE.
                plt.close(fig)

            #if (self.current_epoch % self.show_every) == 0:
            #    plt.show()
            
        except AttributeError:
            n_samples = self.num_classes * self.samples_per_class
            sample_y = np.repeat(np.arange(self.num_classes), self.samples_per_class)
            self.sample_y = torch.LongTensor(sample_y).to(self.device)
            self.fixed_z = torch.FloatTensor(size=(n_samples, self.z_dim)).normal_(0., 1.).to(self.device)
            self.sample()


    def dump_metrics(self):
        
        # D_loss_real
        with open(self.checkpoint_path + 'D_loss_real_{}.pkl'.format(self.current_epoch), 'wb') as f:
            f.write(pickle.dumps(self.d_loss_real))
            
        # D_loss_fake
        with open(self.checkpoint_path + 'D_loss_fake_{}.pkl'.format(self.current_epoch), 'wb') as f:
            f.write(pickle.dumps(self.d_loss_fake))
            
        # total D_loss
        with open(self.checkpoint_path + 'D_loss{}.pkl'.format(self.current_epoch), 'wb') as f:
            f.write(pickle.dumps(self.d_loss))
            
        # G_loss
        with open(self.checkpoint_path + 'G_loss{}.pkl'.format(self.current_epoch), 'wb') as f:
            f.write(pickle.dumps(self.g_loss))
            
        print("Metrics dumped...")
            