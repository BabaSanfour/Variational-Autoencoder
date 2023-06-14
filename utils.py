import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import make_grid, save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def fix_experiment_seed(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def get_dataloaders(data_root, batch_size, image_size):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform = transforms.Compose((
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize))
    
    train = datasets.SVHN(data_root, split='train', download=True, transform=transform)
    test = datasets.SVHN(data_root, split='test', download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    return train_dataloader, test_dataloader

def show_image(image, nrow=8):
  grid_img = make_grid(image.detach().cpu(), nrow=nrow, padding=0)
  plt.imshow(grid_img.permute(1, 2, 0))
  plt.axis('off')

def visualize(data_root, train_batch_size, image_size):
  train_dataloader, _ = get_dataloaders(data_root=data_root, batch_size=train_batch_size, image_size=image_size)
  imgs, labels = next(iter(train_dataloader))

  save_image((imgs + 1.) * 0.5, './results/orig.png')
  show_image((imgs + 1.) * 0.5)

def plot_qe(vector, label):
  fig, ax = plt.subplots(figsize=(12, 8))
  plt.bar(range(len(vector)), vector, label = "Class" + str(label.numpy()))
  plt.title(f"Latent variable distribution for class {label}")
  plt.xlabel("Latent variable index")
  plt.ylabel("Latent variable value")

  plt.legend()
  plt.show()
  plt.savefig(f'./results/qe_{label}.png')

def interpolate(model, z_1, z_2, n_samples, device):
    """  
        Interpolate between z_1 and z_2 with n_samples number of points, with the first point being z_1 and last being z_2.
        
        Parameters
        ----------
        z_1: The first point in the latent space
        z_2: The second point in the latent space
        n_samples: Number of points interpolated

        Returns
        -------
        sample: The mode of the distribution obtained by decoding each point in the latent space
        Size (n_samples, 3, 32, 32)
    """
    lengths = torch.linspace(0., 1., n_samples).unsqueeze(1).to(device)
    z = z_1 + lengths * (z_2 - z_1) 
    return model.decode(z).mode()



