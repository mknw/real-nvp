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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

from scipy import stats  as distatis
from lockandload import * #select_model, mark_version, load_network, track_z, label_zs

def main(args, model_meta_stuff = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
    print("evaluating on: %s" % device)

    # select model.
    if not model_meta_stuff:
        fp_model_root, fp_model, fp_vmarker = select_model(args.root_dir, args.version, test=i)
    else:
        fp_model_root, fp_model, fp_vmarker = model_meta_stuff

    mark_version(args.version, fp_vmarker) # echo '\nV-' >> fp_vmarker

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
    # reminder: type(epoch) == str
    # import ipdb; ipdb.set_trace()
    z, y = label_zs(stats['z'])
    # p-vals, KS p-vals, mu, sigma, skewness, kurtosis, quartiles (.1, .25, .5, .75, .9)
    p, ksp, m, s, b1, b2, qs = distribution_momenta(z) # for each instance. 

    mng_gauss_meas(p, ksp, args.pgaussfile, n_epoch=epoch)

    plot_kde(z, y, np.concatenate(stats['x'], axis=0),
               n_hists=3, filename=fp_distr + '/gauss_hist.png', n_epoch=epoch)

    mark_version(args.version, fp_vmarker, finish=True) # echo '0.2' >> fp_vmarker
    cleanup_version_f(fp_vmarker)


def mng_gauss_meas(p_vals, ks_p_vals, filename='g_test.txt', n_epoch=999):
    ''' to run with version specifier "V-G.0", 
    in order to avoid re-running previous analyses'''
    # check threshold, and write to file. 
    thresholds = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    p_all = 0
    for t in thresholds:
        if np.any(p_vals > t): 
            continue
        else:
            p_all = t
    p_max = np.max(p_vals)
    p_avg = np.mean(p_vals)
    p_min = np.min(p_vals)
    
    ks_p_min = np.min(ks_p_vals)
    ks_p_avg = np.mean(ks_p_vals)
    ks_p_max = np.max(ks_p_vals)
    
    first_ = not os.path.isfile(filename)
    with open(filename, 'a') as f:
        if first_:
            f.write('{},{},{},{},{},{},{}\n'.format('epoch', 'p_avg', 'p_min', 'p_max',
                                              'ks_p_avg', 'ks_p_min', 'ks_p_max'))
            # write headers
        f.write('{},{},{},{},{},{}\n'.format(n_epoch, p_avg, p_min, p_max,
                                         ks_p_avg, ks_p_min, ks_p_max))


def plot_pvals(filename, log_fp='test_gAUSs.txt'):
    import pandas as pd
    fig, ax = plt.subplots(figsize=(10, 7))
    df = pd.read_csv(log_fp)
    x_ax = [i for i in range(120, 690, 10)]
    
    # import ipdb; ipdb.set_trace()
    df.plot(x='epoch', y='p_avg', kind='scatter', color='#cb4b16', ax=ax,
            logy=True, rot=5, fontsize=8, label='p-value')
    df.plot(x='epoch', y='ks_p_avg', kind='scatter', color='#2aa198', ax=ax,
            logy=True, label='KS p-value' )

    ax.set_xticks(range(120, 690, 10), minor=True)
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.5)
    plt.ylabel("$p$ under H0")
    plt.title("Gaussianity test")
    plt.savefig(filename)
    plt.close()
    
    
def distribution_momenta(z, quantiles=[0.1,0.25, 0.5, 0.75,0.9], mean=False, idx=None,
                           ks=True):
    if mean and idx is not None:
        raise ValueError('Please only mean or idx.')

    # last_dim = len(z.shape) - 1
    _, p_vals = distatis.normaltest(z, axis=1)
    # being non-parametric, can be used later with different z\s
    # kolmogorov smirnov takes time: flag it. 
    if ks:
        _, ks_p_vals = ndim_kstest(z, axis=1)

    mus = np.mean(z, axis=1)
    sigmas = np.std(z, axis=1)
    k_vals = distatis.kurtosis(z, axis=1)
    s_vals = distatis.skew(z, axis=1)

    # import ipdb; ipdb.set_trace()
    quartiles  = np.quantile(z, quantiles, axis=1, overwrite_input=True)

    if ks:
        return (p_vals,ks_p_vals,mus,sigmas,s_vals,k_vals,quartiles)
    else:
        return (p_vals,mus,sigmas,s_vals,k_vals,quartiles)





def ndim_kstest(z, axis=1, cdf='norm', args=()):
    z = np.apply_along_axis(distatis.kstest, axis, z, cdf)
    return (z[:,0], z[:,1])

def plot_kde(z, y, x, n_hists=3, filename='sampled_densities.png', n_epoch=99):

    ncols = 10 # nrows = n_hists
    fig, axs = plt.subplots(nrows=n_hists, ncols=ncols, 
                                sharex='all', sharey='all', figsize=(16, 4))
    digit = 0
    plt.title('Sampled z for epoch {}'.format(n_epoch))

    for col in range(ncols):
        digit_idx = np.argwhere(y == digit)
        random_idcs= [np.random.choice(digit_idx.flatten(), replace=False)
                                                              for i in range(n_hists)]
        z_subset = z[random_idcs]
        x_subset = x[random_idcs]
        # use random idcs to feed z[idcs] to `distribution_momenta(...)`
        p_vals, mus, sigmas, s_vals, k_vals, qs = distribution_momenta(z_subset, ks=False)

        for row in range(n_hists):
            sns.distplot(z_subset[row], bins=20, rug=False, kde_kws=dict(linewidth=0.8),
                                        color='C'+str(digit), ax=axs[row, col])
            # axs[row, col].text(-4, .4,
            #                    '$\mu$: {:.2f}\n'
            #                    '$\sigma$: {:.2f}\n'.format(
            #                    mus[row], sigmas[row]),
            axs[row, col].set_xlabel('$\mu$: {:.2f}, $\sigma$: {:.2f}\np-val: {:.5f}'.format(
                                         mus[row], sigmas[row], p_vals[row]),
                               # '$\\beta_1$: {:.3f}\n'
                               # '$\\beta_2$: {:.3f}' s_vals[row], k_vals[row]),
                                     fontsize=7) #, linespacing=0.75)
            maxy_dist = np.max(z_subset[row])
            miny, maxy = axs[row, col].get_ylim()
            minx, maxx = axs[row, col].get_xlim()

            # show handwritten sign-digit for each distribution. 
            x_imagebox = OffsetImage(x_subset[row].reshape(28, 28), zoom=0.5)
            x_imagebox.image.axes = axs[row, col]
            x_ab = AnnotationBbox(x_imagebox, xy=(1, 0.4), 
                              xycoords='data', boxcoords=("axes fraction", "data"))
            axs[row, col].add_artist(x_ab)

            # if row == 0:
            # 	axs[row, col].title.set_text(f'Digit {digit + 1}')

            for q in qs[:,row]:
                axs[row, col].axvline(x=q, ymin=0,ymax=maxy_dist, c='red', alpha=0.3, linewidth=0.5)


        digit += 1
    # plt.tight_layout()
    plt.setp(axs, xlim=(-6, 6)) # ylim seems to be fine. 
    plt.subplots_adjust(wspace=0.5, hspace=0.5, bottom=0.15)
    plt.savefig(filename)
    plt.close()

    # with open(filename, 'w') as f:
    # 	f.write(

def track_xz(net, loader, device, loss_fn, **kwargs):

    net.eval()
    loss_meter = util.AverageMeter()
    bpd_meter = util.AverageMeter()
    
    with tqdm(total=len(loader.dataset)) as progress:

        digits_stds = [np.array([]).reshape(0, 1) for i in range(10)]
        digits_means = [np.array([]).reshape(0, 1) for i in range(10)]

        ''' not quite but useful for reference. '''
        digits_z = [np.array([]).reshape(0, 1, 28, 28) for i in range(10)]
        digits_x = [np.array([]).reshape(0, 1, 28, 28) for i in range(10)]

        for x, y in loader:
            x = x.to(device)
            z, sldj = net(x, reverse=False)
            mean_z = z.mean(dim=[1, 2, 3]) # `1` is channel.
            std_z = z.std(dim=[1, 2, 3])

            for n in range(10):
                # check if concatenation axis needs to be specified for ZZZZ.
                digit_indices = (y==n).nonzero()
                digit_indices_1dim = (y==n).nonzero(as_tuple=True)
                dig_std_z = std_z[ digit_indices ]
                dig_mean_z = mean_z[ digit_indices ]
                z_dig = z[digit_indices_1dim]
                x_dig = x[digit_indices_1dim]
                # here we concatenate each 'array' to the previous one (or initialized arr. of size=(0, 1)).
                digits_stds[n] = np.concatenate(( digits_stds[n], dig_std_z.to('cpu').detach().numpy() ))
                digits_means[n] = np.concatenate(( digits_means[n], dig_mean_z.to('cpu').detach().numpy() ))
                # concatenate whole z space. 
                digits_z[n] = np.concatenate(( digits_z[n], z_dig.to('cpu').detach().numpy() ))
                digits_x[n] = np.concatenate(( digits_x[n], x_dig.to('cpu').detach().numpy() ))

            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            bpd_meter.update(util.bits_per_dim(x, loss_meter.avg), x.size(0))

            progress.set_postfix(loss=loss_meter.avg,
                                         bpd=bpd_meter.avg)
            progress.update(x.size(0))
    # add loss.avg() and bpd.avg() somewhere in the plot. 
    
    return {'std': digits_stds, 'mean': digits_means,
            'z': digits_z, 'x': digits_x}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--dataset', '-ds', default='mnist', type=str, help='MNIST or CIFAR-10')
    # parser.add_argument('--num_samples', default=121, type=int, help='Number of samples at test time')
    # XXX PARAMS XXX
    version_ = 'V-G.5'
    force_ = False
    # Analysis
    parser.add_argument('--force', '-f', action='store_true', default=force_, help='Re-run z-space anal-yses.')
    parser.add_argument('--version', '-v', default=version_, type=str, help='Analyses iteration')

    # General architecture parameters
    # XXX PARAMS XXX
    pgaussfile_ = 'test_gAUSs.txt'
    pgaussfile_ = 'dmnist_gauss.txt' # for testing purposes
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
    

    for i in range(120, 690, 10):
        model_meta_stuff = select_model(root_dir_, version_, test=i)
        main(parser.parse_args(), model_meta_stuff)
        print(" done.")
    plot_pvals('figs/dmnist_gauss.png', pgaussfile_)

