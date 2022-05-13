import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import lambertw

import torch
from torch.autograd import grad


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr_scheduler(optimizer, args):
    """Learning Rate Scheduler"""
    if args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_every, threshold=0.001, patience=1)
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=0)
    else:
        raise NotImplementedError

    return scheduler


def get_gradient_penalty(discriminator, real_images, fake_images, device, eps = 1e-12):
    """Gradient Penalty"""
    epsilon = torch.rand(real_images.size(0), 1, 1, 1).to(device)
    x_hat = (epsilon * real_images + (1 - epsilon) * fake_images).requires_grad_(True)

    x_hat_prob = discriminator(x_hat)
    x_hat_grad = torch.ones(x_hat_prob.size()).to(device)

    gradients = grad(outputs=x_hat_prob, inputs=x_hat, grad_outputs=x_hat_grad, create_graph=True,
                     retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = torch.sqrt(torch.sum(gradients ** 2, dim=1) + eps)
    gradient_penalty = torch.mean((gradient_penalty-1)**2)

    return gradient_penalty


def plot_series(series, fake_series, generator, epoch, args, path):
    """Plot Samples"""
    plt.figure(figsize=(10, 5))
    plt.plot(series, label='real')
    plt.plot(fake_series, label='fake')
    plt.grid(True)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel(args.column)
    plt.title('Fake_{}_using{}_and_{}_Epoch{}.png'.format(args.dataset_name, generator.__class__.__name__, args.criterion.upper(), epoch+1))
    plt.savefig(os.path.join(path, 'Fake_{}_using{}_and_{}_Epoch{}.png'.format(args.dataset_name, generator.__class__.__name__, args.criterion.upper(), epoch+1)))
    plt.close("all")

def make_csv(series, fake_series, generator, epoch, args, path):
    """Convert to CSV files"""
    data = pd.DataFrame({'real': series, 'fake': fake_series})
    data.to_csv(os.path.join(path, 'Fake_{}_using{}_and_{}_Epoch{}.csv'.format(args.dataset_name, generator.__class__.__name__, args.criterion.upper(), epoch+1)), index=False)
