#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

#/var/scratch/mao540/miniconda3/envs/sobab37/bin/python

import torch 
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import util
import argparse
import os
import numpy as np
from models import RealNVP, RealNVPLoss
from tqdm import tqdm
from random import randrange
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

from scipy import stats  as distatis
from mnist_simil import * #select_model, mark_version, load_network, track_z, label_zs

def main(args, model_meta_stuff = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
    print("evaluating on: %s" % device)

    # select model.
    if not model_meta_stuff:
        fp_model_root, fp_model, fp_vmarker = select_model(args.root_dir, args.version, test=i)
    else:
        fp_model_root, fp_model, fp_vmarker = model_meta_stuff

    # mark_version(args.version, fp_vmarker) # echo '\nV-' >> fp_vmarker

    epoch = fp_model_root.split('_')[-1]
    ''' stats filenames memo: '''
    # 1: '/z_mean_std.pkl' # XXX obsolete! XXX
    # 2: '/latent_mean_std_Z.pkl'
    stats_filename = fp_model_root + '/zlatent_mean_std_Z.pkl'
    if os.path.isfile(stats_filename) and not args.force:
        print('Found cached file, skipping computations of mean and std.')
        stats = torch.load(stats_filename)
    else:
        if args.dataset == 'mnist':
            transform_train = transforms.Compose([
                transforms.ToTensor()
                # transforms.ColorJitter(brightness=0.3)
            ])
            #torchvision.transforms.Normalize((0.1307,), (0.3081,)) # mean, std, inplace=False.
            transform_test = transforms.Compose([
                transforms.ToTensor()
            ])
            # trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
            # trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_test)
            testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            print('Building model..') 

        elif args.dataset == 'CIFAR-10':
            # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            #torchvision.transforms.Normalize((0.1307,), (0.3081,)) # mean, std, inplace=False.
            transform_test = transforms.Compose([
                transforms.ToTensor()
            ])
            trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
            trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
            testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            print('Building model..')
            net = RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)

        net = load_network( fp_model, device, args)
        loss_fn = RealNVPLoss()
        # stats = track_distribution(net, testloader, device, loss_fn)
        stats = track_xz(net, testloader, device, loss_fn)
        torch.save(stats, stats_filename)

    fp_distr = fp_model_root + '/distr' # distributions
    fp_simil = fp_model_root + '/similarity' # similarity

    # reminder: type(epoch) == str
    # import ipdb; ipdb.set_trace()
    z, y = label_zs(stats['z'])

    distances_all = calculate_distance(stats, joint=True)
    distances_std = calculate_distance(stats, measure='mean')
    distances_mu = calculate_distance(stats, measure='std')

    dct_sim_mtx = dct_similarity( x_vector_dct(stats['x'], y) )

    x_labels = ['Euclidian distance ($\mu, \sigma$)',
                'Euclidian distance ($\mu$)',
                'Euclidian distance ($\sigma$)']
    plot_names = ['/regr_allmeas_dct.png', '/regr_moo_dct.png', '/regr_std_dct.png']
    filenames = [fp_simil + pn for pn in plot_names]
    momenta_sims = [distances_all, distances_mu, distances_std]


    plot_regr(momenta_sims, dct_sim_mtx, x_labels, fp_simil + '/regr_sub.png')


    # import ipdb; ipdb.set_trace()

    heatmap(dct_sim_mtx, filename = fp_simil + '/dct_sim.png',
            plot_title = 'Average distance in fourier space (DCT)')

    mark_version(args.version, fp_vmarker, finish=True) # echo '0.2' >> fp_vmarker
    cleanup_version_f(fp_vmarker)

def purify_mtx(x):
    x = np.tril(x, k=-1)
    return x[x!=0].flatten()



def plot_regr(dist_moms, dist_dct, x_labels, filename):

    fig, axs = plt.subplots(1,3, figsize=(15, 5))
    
    dist_moms = [purify_mtx(y) for y in dist_moms]
    dist_dct = purify_mtx(dist_dct)
    for i in range(3):
        m, b, r, p, se = distatis.linregress(dist_moms[i], dist_dct)

        axs[i].plot(dist_moms[i], dist_dct, 'ob', markersize=2)
        
        if i == 2:
            axs[i].plot(dist_moms[i], b + m * dist_moms[i], '-r', markersize=3)
        else:
            axs[i].plot(dist_moms[i], b + m * dist_moms[i], '-k', markersize=3)
        # x axis
        axs[i].set_xlabel(x_labels[i])
        # y axis
        axs[i].set_ylabel('$x$ dct distance')
        extra = mpl.patches.Circle((0, 0), radius=1, fc="w", fill=False, edgecolor='none', linewidth=0)
        axs[i].legend([extra, extra, extra],
                      ('$R^2: {:.3f}$'.format(r**2),
                      'r: {:.3f}'.format(r),
                      'p-val: {:.3f}'.format(p) ))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def regr_distances(z_dist, dct_dist):
    z_dist = purify_mtx(z_dist)
    dct_dist = purify_mtx(dct_dist)
    return distatis.linregress(z_dist, dct_dist)


def dct2(x):
    from scipy.fftpack import dct
    return dct(
           dct(x, axis=-1, norm='ortho'), 
                  axis=-2, norm='ortho')


def x_vector_dct(x_s, y):
    # x: list of arrays
    # y: array
    print('we are at the function')

    dct_shape = [10] + list(x_s[0].shape[-2:])

    xs_dct_avg = np.zeros(shape=dct_shape)

    for idx, x_dig in enumerate(x_s):
        x_vec_dct = dct2(x_dig)
        xs_dct_avg[idx] = np.mean(x_vec_dct, axis=0)

    return xs_dct_avg

def dct_similarity(dct_vec):
    distances = np.zeros(shape=(10, 10))
    for i in range(dct_vec.shape[0]):
        for j in range(dct_vec.shape[0]):
            distance = np.linalg.norm(dct_vec[i] - dct_vec[j])
            distances[i, j] = distance
    return distances





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--dataset', '-ds', default='mnist', type=str, help='MNIST or CIFAR-10')
    # parser.add_argument('--num_samples', default=121, type=int, help='Number of samples at test time')
    # XXX PARAMS XXX
    version_ = 'V-G.6'
    force_ = False
    # Analysis
    parser.add_argument('--force', '-f', action='store_true', default=force_, help='Re-run z-space analyses.')
    parser.add_argument('--version', '-v', default=version_, type=str, help='Analyses iteration')

    # General architecture parameters
    # XXX PARAMS XXX
    pgaussfile_ = 'test_gAUSs.txt'
    pgaussfile_ = 'dmnist_gauss_iso.txt' # for testing purposes
    net_ = 'densenet'
    dataset_ = 'mnist'
    if dataset_ == 'mnist':
        in_channels_ = 1
        num_samples_ = 121
    elif dataset_ == 'celeba':
        in_channels_ = 3
        # num_samples_ = 64

    if net_ == 'resnet':
        root_dir_ = 'data/res_3-8-32'
        batch_size_ = 256

        num_scales_ = 3
        mid_channels_ = 32
        num_levels_ = 8

    if net_ == 'densenet':
        # train one epoch: 6:14
        # test: 12:12
        num_scales_ = 3
        if dataset_ == 'celeba':
            batch_size_ = 64
            mid_channels_ = 128
            num_levels_ = 8
            num_samples_ = 64
            if gpus_ == '[0, 1]':
                batch_size_ = 16
                num_samples_ = 16

        elif dataset_ == 'mnist': # data/dense_test6
            root_dir_ = 'data/dense_test6'
            batch_size_ = 218
            mid_channels_ = 120
            num_levels_ = 10
            num_samples_ = 121
            num_scales_ = 3

    parser.add_argument('--net_type', default=net_, help='CNN architecture (resnet or densenet)')
    parser.add_argument('--batch_size', default=batch_size_, type=int, help='Batch size')
    parser.add_argument('--root_dir', default=root_dir_, help='Analyses root directory.')
    parser.add_argument('--num_scales', default=num_scales_, type=int, help='Real NVP multi-scale arch. recursions')
    parser.add_argument('--in_channels', default=in_channels_, type=int, help='dimensionality along Channels')
    parser.add_argument('--mid_channels', default=mid_channels_, type=int, help='N of feature maps for first resnet layer')
    parser.add_argument('--num_levels', default=num_levels_, type=int, help='N of residual blocks in resnet')
    parser.add_argument('--pgaussfile', default=pgaussfile_, type=str, help='log gaussianity to file')
    

    for i in range(120, 680, 10):
        if i in [640, 670]:
            continue
        model_meta_stuff = select_model(root_dir_, version_, test=i)
        main(parser.parse_args(), model_meta_stuff)
    # 	print(" done.")
    # plot_pvals('figs/dmnist_gauss_iso.png', pgaussfile_)

