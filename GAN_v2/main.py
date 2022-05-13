import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os, sys
import argparse
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from dataset import ReadDataset

from utils import make_dirs, get_lr_scheduler
from utils import get_gradient_penalty, plot_series, make_csv

from tensorboardX import SummaryWriter



def main(args):

    ## log file
    os.makedirs(args.log_path, exist_ok=True)
    log_writer = SummaryWriter(logdir=args.log_path)

    ## training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## random number generator
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    ## saving directory
    paths = [args.samples_path, args.weights_path, args.csv_path]
    for path in paths:
        make_dirs(path)

    ## loading data and batching
    dataset = ReadDataset(root=args.data_path, classname=args.dataset_name)
    print("dataset num: ", dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    numBatch = len(dataloader)
    print("dataloader num: ", numBatch)

    ## initialize the model
    if args.model == 'conv':
        from models_gan import Generator, Discriminator
        D = Discriminator(in_channels=1, mid_dim=64, out_channel=1).to(device)
        G = Generator(in_channels=1, out_channels=1).to(device)
    elif args.model == 'lstm':
        from models import LSTMGenerator, LSTMDiscriminator
        D = LSTMDiscriminator(ts_dim=800).to(device)
        G = LSTMGenerator(latent_dim=50, ts_dim=800).to(device)
    elif args.model == 'fnn':
        from models import NetGenerator, NetDiscriminator
        D = NetDiscriminator.to(device)
        G = NetGenerator.to(device)
    else:
        print("args.model not exists. only 'conv' or 'lstm' .")
        raise NotImplementedError

    ## initialize the loss function
    if args.criterion == 'l2':
        criterion = nn.MSELoss()
    elif args.criterion == 'wgangp':
        pass
    else:
        raise NotImplementedError

    ## optimizer
    if args.optim == 'sgd':
        D_optim = torch.optim.SGD(D.parameters(), lr=args.lr, momentum=0.9)
        G_optim = torch.optim.SGD(G.parameters(), lr=args.lr, momentum=0.9)

    elif args.optim == 'adam':
        D_optim = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0., 0.9))
        G_optim = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0., 0.9))

    else:
        raise NotImplementedError

    ## model scheduler
    D_optim_scheduler = get_lr_scheduler(D_optim, args)
    G_optim_scheduler = get_lr_scheduler(G_optim, args)

    ## initialize the loss
    D_losses, G_losses = list(), list()
    print("Training Time Series GAN started with total epoch of {}.".format(args.num_epochs))

    ## training loop function
    for epoch in range(args.num_epochs):
        for i, series in tqdm(enumerate(dataloader), desc='Training'):
            series = series.float().to(device)
            if args.model == 'lstm':
                series = series.transpose(0, 1)

            G_optim.zero_grad()
            D_optim.zero_grad()

            #######################
            # training the Discriminator #
            #######################
            if args.criterion == 'l2':
                n_critics = 1
            elif args.criterion == 'wgangp':
                n_critics = 2
            else:
                n_critics = 1

            for j in range(n_critics):
                ## random noise
                if args.model == 'conv':
                    noise = torch.randn(series.size(0), args.latent_dim).to(device)
                elif args.model == 'lstm':
                    noise = torch.randn(series.size(1), args.latent_dim).to(device)
                    noise = torch.unsqueeze(noise, dim=0)

                ## true or false
                prob_real = D(series)
                ## calculatiing the loss function
                if args.criterion == 'l2':
                    real_labels = torch.ones(prob_real.size()).to(device)
                    D_real_loss = criterion(prob_real, real_labels)
                elif args.criterion == 'wgangp':
                    D_real_loss = -torch.mean(prob_real)

                ##
                if args.model == 'conv':
                    fake_series = G(series, noise)
                elif args.model == 'lstm':
                    fake_series = G(noise)

                prob_fake = D(fake_series.detach())

                if args.criterion == 'l2':
                    fake_labels = torch.zeros(prob_fake.size()).to(device)
                    D_fake_loss = criterion(prob_fake, fake_labels)
                elif args.criterion == 'wgangp':
                    D_fake_loss = torch.mean(prob_fake)
                    D_gp_loss = args.lambda_gp * get_gradient_penalty(D, series, fake_series, device)

                ## Discriminator loss
                D_loss = D_fake_loss + D_real_loss

                if args.criterion == 'wgangp':
                    D_loss += args.lambda_gp * D_gp_loss

                ## logging
                D_loss.backward()
                D_optim.step()

            ###################
            # training generator #
            ###################

            ## intialize the model
            if args.model == 'conv':
                fake_series = G(series, noise)
            elif args.model == 'lstm':
                fake_series = G(noise)

            prob_fake = D(fake_series)

            ## calculatiing the loss
            if args.criterion == 'l2':
                real_labels = torch.ones(prob_fake.size()).to(device)
                G_loss = criterion(prob_fake, real_labels)

            elif args.criterion == 'wgangp':
                G_loss = -torch.mean(prob_fake)

            ##
            G_loss.backward()
            G_optim.step()

            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            ## updating the learning rate
            D_optim_scheduler.step()
            G_optim_scheduler.step()

            ## output the training result
            if (i+1) % args.log_every == 0:
                log_writer.add_scalar('loss_g', np.average(G_losses), i+epoch*numBatch)
                log_writer.add_scalar('loss_d', np.average(D_losses), i+epoch*numBatch)

                # Print Statistics and Save Model #
                print("Epochs [{}/{}] [{}/{}] | D Loss {:.4f} | G Loss {:.4f}".format(epoch+1, args.num_epochs, i+1, numBatch, np.average(D_losses), np.average(G_losses)))
                torch.save(G.state_dict(), os.path.join(args.weights_path, 'TS_{}_using{}_and_{}_Epoch_{}.pth'.format(args.dataset_name, G.__class__.__name__, args.criterion.upper(), epoch + 1)))

                ## generating the example
                if args.model == 'conv':
                    fixed_noise = torch.randn(series.size(0), args.latent_dim).to(device)
                    fake_series = G(series, fixed_noise)
                elif args.model == 'lstm':
                    fixed_noise = torch.randn(series.size(1), args.latent_dim).to(device)
                    fixed_noise = torch.unsqueeze(fixed_noise, dim=0)
                    fake_series = G(fixed_noise)

                series_ele = series[0].cpu().data.numpy()[0]
                fake_series_ele = fake_series[0].cpu().data.numpy()[0]

                print(series_ele.shape, fake_series_ele.shape)
                plot_series(series_ele, fake_series_ele, G, epoch, args, args.samples_path)
                make_csv(series_ele, fake_series_ele, G, epoch, args, args.csv_path)

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_num', type=int, default=0, help='gpu number')
    parser.add_argument('--seed', type=int, default=7777, help='seed')

    parser.add_argument('--data_path', type=str, default='./dataset_npy', help='data path')
    parser.add_argument('--column', type=str, default='Appliances', help='which column to generate')

    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='total epoch for training')
    parser.add_argument('--log_every', type=int, default=2, help='save log data for every default iteration')

    parser.add_argument('--model', type=str, default='conv', choices=['conv', 'lstm', 'fnn'], help='which network to train')
    parser.add_argument('--dataset_name', type=str, default='normal', choices=['normal', 'fault'], help='which dataset to train')

    parser.add_argument('--delta', type=float, default=0.7, help='delta')
    parser.add_argument('--constant', type=float, default=0, help='If zero in the original data, please set it as non-zero, e.g. 1e-1')
    parser.add_argument('--latent_dim', type=int, default=50, help='noise dimension')

    parser.add_argument('--criterion', type=str, default='l2', choices=['l2', 'wgangp'], help='criterion')
    parser.add_argument('--lambda_gp', type=int, default=10, help='constant for gradient penalty')

    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'], help='which optimizer to update')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay learning rate')
    parser.add_argument('--lr_decay_every', type=int, default=1000, help='decay learning rate for every default epoch')
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'plateau', 'cosine'], help='learning rate scheduler')

    parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
    parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
    parser.add_argument('--csv_path', type=str, default='./results/csv/', help='csv path')
    parser.add_argument('--log_path', type=str, default='./results/logs/', help='log dir')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    main(args)