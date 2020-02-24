import os
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms


def get_dataloaders(train_root, test_root, batch_size, shuffle_data=True):

    means = [0.5, 0.5, 0.5]
    stds = [0.5, 0.5, 0.5]

    train_transformations = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, 
                             std=means)
    ])

    test_transformations = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, 
                             std=stds)
    ])

    train_dataset = datasets.ImageFolder(root=train_root, transform=train_transformations)

    test_dataset = datasets.ImageFolder(root=test_root, transform=test_transformations)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                             shuffle=shuffle_data, num_workers=4, 
                                             pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=4, 
                                             pin_memory=shuffle_data)
    
    return train_dataloader, test_dataloader


def make_grid(tensor, samples_per_class, num_classes=8, padding=5, im_size=256):
    
    """
    TODO: Write documentation
    """
    
    n_cols = samples_per_class
    n_rows = num_classes
    
    # Each row represets a class
    images = np.transpose(tensor.numpy(), (0, 2, 3, 1))
    images = np.clip((images * 127.5) + 127.5, 0., 255.)
    
    grid = np.zeros((im_size*n_rows + padding*(n_rows-1), im_size*n_cols + padding*(n_cols-1), 3))
    
    row_ptr = 0
    for i in range(0, images.shape[0], n_cols):
        col_ptr = 0
        for j in range(n_cols):
            grid[row_ptr:row_ptr+im_size, col_ptr:col_ptr+im_size, :] = images[i+j, :, :, :]
            col_ptr += im_size + padding
            col_ptr = min(grid.shape[1], col_ptr)
        row_ptr += im_size + padding
        row_ptr = min(grid.shape[0], row_ptr)
        
    return grid.astype(np.uint8)


def create_interpolation(filename, im_path, fps=15):
    
    """
    Creates an interpolation gif using all the images dumped at the end of every X epochs.
    
    Parameters:
    filename: the output file URI. Ensure that it is a GIF file.
    im_path: the path where the images are dumped.
    fps: tuneable fps setting for the gif.
    
    Returns:
    None
    """
    
    # Read the image names of the files and wrap in a np array.
    images = np.array([x for x in os.listdir(im_path) if str(x).endswith('.png')])
    
    # Create an auxiliary list that contains the parsed filenames. 
    int_indices = [int(str(x).split('.')[0]) for x in os.listdir(im_path) if str(x).endswith('.png')]
    
    # Sort the auxiliary list and return the indices.
    sorted_int_indices = np.argsort(int_indices)
    
    # Sort the OG list using these indices and convert back to list.
    images = images[sorted_int_indices].tolist()
    
    # Buffer for np arrays.
    im_buffer = []
    
    for im in images:
        # Keep the handle so as to manually close it and avoid
        # memory leaks.
        _im = Image.open(im_path + im, 'r')
        
        # Note: W*H*3 is transposed to H*W*3 here.
        im_np = np.array(_im)
        
        # Append to the buffer.
        im_buffer.append(im_np)
        
        # Close the stream.
        _im.close()
    
    # Finally, write the sequence of np arrays to a gif.
    imageio.mimwrite(filename, im_buffer, fps=fps)
    
    
def plot_losses(filename, g_loss, d_loss, superimpose=False):
    
    ncols = 1 if superimpose else 2
    
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(10, 5))
    
    if superimpose:
        ax.plot(g_loss, 'r-')
        ax.plot(d_loss, 'b-')
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Losses')
        ax.legend(['Generator loss', 'Discriminator loss'], loc='upper right')
        ax.set_title("Loss plots for the Generator & the Discriminator")
    else:
        ax[0].plot(g_loss, 'r-')
        ax[1].plot(d_loss, 'b-')

        ax[0].legend(['Generator loss'], loc='upper right')
        ax[0].set_title("Loss plots for the Generator")
        ax[0].set_xlabel('Iterations')
        ax[0].set_ylabel('Loss')
        
        ax[1].legend(['Discriminator loss'], loc='upper right')
        ax[1].set_title("Loss plots for the Discriminator")
        ax[1].set_xlabel('Iterations')
        ax[1].set_ylabel('Loss')
        
    plt.savefig(filename, bbox_inches='tight')
    
    plt.show()