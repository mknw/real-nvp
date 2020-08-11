#!/var/scratch/mao540/miniconda3/envs/sobab37/bin/python

import argparse
import os
import torch 
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import util

from models import RealNVP, RealNVPLoss
from tqdm import tqdm


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
    print("training on: %s" % device)
    start_epoch = 0

    if args.dataset == 'mnist':
        from load_dataset import load_mnist
        trainloader = load_mnist()
        testloader = load_mnist(test=True)

    elif args.dataset.lower() == 'celeba':
        from load_dataset import load_celeba
        trainloader = load_celeba(args.batch_size, args.img_size)
        testloader = load_celeba(args.batch_size, args.img_size, test=True)

    elif args.dataset == 'CIFAR-10':
        from load_dataset import load_cifar10
        trainloader = load_cifar10()
        testloader = load_cifar10(test=True)
    else:
        raise ValueError('Datasets: `MNIST`, `CelebA` or `CIFAR-10`.')

    net = RealNVP( **filter_args(args.__dict__) )
    net = net.to(device)

    if str(device).startswith('cuda'):
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark
    import ipdb; ipdb.set_trace()

    # Load checkpoint.
    print('Resuming from checkpoint at ' + args.model_dir + '/model.pth.tar...')
    assert os.path.isdir(args.model_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.model_dir + '/model.pth.tar')
    net.load_state_dict(checkpoint['net'])
    global best_loss
    try:
        best_loss = checkpoint['test_loss']
    except:
        best_loss = checkpoint['best_loss']
    # we start epoch after the saved one (avoids overwrites).
    epoch = checkpoint['epoch']

    # loss_fn = RealNVPLoss()
    # param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    # optimizer = optim.Adam(param_groups, lr=args.lr, eps=1e-7)

    # for epoch in range(start_epoch, start_epoch + args.num_epochs):
    test_settings = filter_args(args.__dict__, fields = ['num_samples', 'dir_samples',
                                                           'in_channels', 'img_size', 'temp'])
    for t in [0.6, 0.65,0.7, 0.75, 0.8, .85, .9, 0.95]:
        test_settings['temp'] = t
        test(epoch, net, testloader, device, **test_settings)



def sample(net, num_samples, in_channels, device, img_size, temp=0.8):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    
    z = torch.randn((num_samples, in_channels, img_size, img_size), dtype=torch.float32, device=device) #changed 3 -> 1
    x, _ = net(z * temp, reverse=True)
    x = torch.sigmoid(x)
    return x, z


def test(epoch, net, testloader, device, **args):
    global best_loss
    net.eval()
    save_dir = args['dir_samples'] + '/epoch_{:03d}'.format(epoch) #  + str(epoch)
    os.makedirs(save_dir, exist_ok=True)


    # import ipdb; ipdb.set_trace()
    sample_fields = ['num_samples', 'in_channels', 'img_size', 'temp']
    images, latent_z = sample(net, device=device, **filter_args( args, fields=sample_fields ) )

    # plot x and z
    num_samples = args['num_samples']
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    t = args['temp']
    torchvision.utils.save_image(images_concat, save_dir+f"/x{t}.png")
    print(f'saved to {save_dir}/x{t}.png')



def filter_args(arg_dict, fields=None):
    """only pass to network architecture relevant fields."""
    if not fields:
        # if arg_dict['net_type'] == 'resnet':
        fields = ['net_type', 'num_scales', 'in_channels', 'mid_channels', 'num_levels']
        # elif arch['net_type'] == 'densenet':
        # 	arch_fields = ['net_type', 'num_scales', 'in_channels', 'mid_channels', 'depth']
    return {k:arg_dict[k] for k in fields if k in arg_dict}


class GaussianNoise(object):

    def __init__(self, mean=0., std=.1, restrict_range=True):
        self.std = std
        self.mean = mean
        self.restrict_range = restrict_range

    def __call__(self, tensor):
        tensor += torch.randn(tensor.size()) * self.std + self.mean
        if self.restrict_range:
            return tensor.clamp(1e-8, 1)
        else:
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



class Normie(object):
    '''class for normies'''
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, tensor):
        tensor -= tensor.min()
        tensor /= tensor.max()
        return tensor

def find_last_epoch_model(fp):
    dirs_l = os.listdir(fp)
    dirs_e = [d for d in dirs_l if d.startswith('epoch_') 
                                     and d[-3:].isdigit()]
    dirs_e.sort()
    last_epoch = dirs_e[-1]
    print('Last model it.: ' + last_epoch)
    return fp + '/' + last_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP training on various datasets.')
    # system
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    # hyperparameters
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate') # changed from 1e-3 for MNIST
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                                            help='L2 regularization (only applied to the weight norm scale factors)')
    parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')

    ## logic for default values. (Would likely need to be determined from command line).

    # 1. Dataset : 'celeba', 'MNIST', 'CIFAR' (not tested)
    # dataset_ = 'mnist' 
    dataset_ = 'celeba'
    # 2. Architecture
    net_ = 'densenet'  # 2.
    # 3. Samples dir_
    dir_ = '/1_' + net_[:3] +'_'+dataset_ if dataset_ == 'celeba' else '/dense_test6' # 3.
    # dir_ = '/dense_test6'
    # only multi-gpu if celeba.resnet.
    # 4. GPUs
    gpus_ = '[0, 1]' if net_ == 'densenet' and dataset_=='mnist'  else '[0]' # 4.
    # 5. resume training?
    resume_ = True # 5.
    # 6. resize 
    temp_ = 0.8

    if dataset_ == 'mnist':
        in_channels_= 1
    elif dataset_ == 'celeba':
        in_channels_= 3
        img_size = 64

    if resume_:
        # model_dir_ = find_last_epoch_model('data' + dir_)
        model_dir_ = 'data/' + dir_ + '/epoch_700'
        
    parser.add_argument('--img_size', default=img_size, type=eval)

    parser.add_argument('--resume', '-r', action='store_true', default=resume_, help='Resume from checkpoint')
    parser.add_argument('--gpu_ids', default=gpus_, type=eval, help='IDs of GPUs to use')
    parser.add_argument('--net_type', default=net_, help='CNN architecture (resnet or densenet)')
    parser.add_argument('--dir_samples', default="data" + dir_ , help="Directory for storing generated samples")

    # dataset
    parser.add_argument('--dataset', '-ds', default=dataset_.lower(), type=str, help="MNIST or CIFAR-10")
    
    parser.add_argument('--in_channels', default=in_channels_, type=int, help='dimensionality along Channels')

    # architecture
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
            batch_size_ = 1024 if len(gpus_) > 3 else 512
            mid_channels_ = 120
            num_levels_ = 10
            num_samples_ = 121
            num_scales_ = 3

    elif net_ == 'resnet':
        if dataset_ == 'celeba': # CelebA 
            batch_size_ = 8 if len(gpus_) > 3 else 1
            mid_channels_ = 32
            num_levels_ = 8
            num_samples_ = 8
            num_scales_ = 2
        elif dataset_ == 'mnist': # data/dense_test6
            batch_size_ = 218
            mid_channels_ = 32
            num_levels_ = 8
            num_samples_ = 121
            num_scales_ = 3
            # time on TitanX-Pascal (batch_size == 8):
            # training : 8:07:44 / epoch
            # testing :  16:55   / epoch
            # time on RTX2080Ti (batch_size == 4):

    parser.add_argument('--temp', default=temp_, type=float, help='Batch size')
    parser.add_argument('--batch_size', default=batch_size_, type=int, help='Batch size')
    parser.add_argument('--mid_channels', default=mid_channels_, type=int, help='N of feature maps for first resnet layer')
    parser.add_argument('--num_levels', default=num_levels_, type=int, help='N of residual blocks in resnet, or N of dense layers in densenet (depth)')
    parser.add_argument('--model_dir', default=model_dir_, help="Directory for storing generated samples")
    parser.add_argument('--num_samples', default=num_samples_, type=int, help='Number of samples at test time')
    parser.add_argument('--num_scales', default=num_scales_, type=int, help='Real NVP multi-scale arch. recursions')

    
    best_loss = 5e5

    main(parser.parse_args())
