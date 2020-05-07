#!/var/scratch/mao540/miniconda3/envs/sobab37/bin/python

import argparse
import os
import torch 
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util

floatAboat = 0.1
from models import RealNVP, RealNVPLoss
from tqdm import tqdm


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
    print("training on: %s" % device)
    start_epoch = 0

    if args.dataset == 'MNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # GaussianNoise(std=0.17) # yeah!
            # transforms.Normalize()
        ])
        #torchvision.transforms.Normalize((0.1307,), (0.3081,)) # mean, std, inplace=False.
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

        trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        print('Building model..') 
        # net = RealNVP( **filter_args(args.__dict__) )

    elif args.dataset.lower() == 'celeba':
        # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(176),
            # TODO: add conditional based on args.resize_hw_
            transforms.Resize(size=args.resize_hw),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(176),
            transforms.Resize(size=args.resize_hw),
            transforms.ToTensor()
        ])
        target_type = ['attr', 'bbox', 'landmarks']
        trainset = torchvision.datasets.CelebA(root='data', split='train', target_type=target_type[0], download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CelebA(root='data', split='test', target_type=target_type[0], download=True, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        print('Building model..')
        # net = RealNVP( **filter_args(args.__dict__) )
        pass

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
        # net = RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
    else:
        raise ValueError('Datasets: `MNIST`, `CelebA` or `CIFAR-10`.')


    net = RealNVP( **filter_args(args.__dict__) )
    net = net.to(device)

    if str(device).startswith('cuda'):
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    if args.resume: # or not args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ' + args.dir_model + '/model.pth.tar...')
        assert os.path.isdir(args.dir_model), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.dir_model + '/model.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        best_loss = checkpoint['test_loss']
        # we start epoch after the saved one (avoids overwrites).
        start_epoch = checkpoint['epoch'] + 1

    loss_fn = RealNVPLoss()
    param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr, eps=1e-7)

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train_stats = train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm)
        test_settings = filter_args(args.__dict__, fields = ['num_samples', 'dir_samples', 'in_channels', 'resize_hw'])
        test_settings = {**test_settings, **train_stats} # merge dicts.
        test(epoch, net, testloader, device, loss_fn, **test_settings)


def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    bpd_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader: # TODO: y -> _
            x = x.to(device)
            optimizer.zero_grad()
            # import ipdb; ipdb.set_trace()
            z, sldj = net(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            bpd_meter.update(util.bits_per_dim(x, loss_meter.avg))
            progress_bar.set_postfix(loss=loss_meter.avg,
                                                                bpd=bpd_meter.avg)
            progress_bar.update(x.size(0))
            #debugopt:
    return {'train_loss': loss_meter.avg, 
             'train_bpd': bpd_meter.avg}

def save_imgrid(tensor, name):
    grid = torchvision.utils.make_grid(tensor, nrow=int(tensor.shape[0] ** 0.5), padding=1, pad_value=255)
    torchvision.utils.save_image(grid, name)
    return

def sample(net, num_samples, in_channels, device, resize_hw=None):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    
    if not resize_hw:
        side_size = 28
    else:
        side_size, side_size = resize_hw
    print("sampling with z space sized: {side_size}x{side_size}.")
    z = torch.randn((num_samples, in_channels, side_size, side_size), dtype=torch.float32, device=device) #changed 3 -> 1
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)
    return x, z


def test(epoch, net, testloader, device, loss_fn, **args):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    bpd_meter = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)+1) as progress_bar:
            for x, _ in testloader:
                x = x.to(device)
                z, sldj = net(x, reverse=False)
                loss = loss_fn(z, sldj)
                loss_meter.update(loss.item(), x.size(0))
                # bits per dimensions
                bpd_meter.update(util.bits_per_dim(x, loss_meter.avg), x.size(0))

                progress_bar.set_postfix(loss=loss_meter.avg,
                                                                 bpd=bpd_meter.avg)
                progress_bar.update(x.size(0))
                
    # Save checkpoint
    save_dir = args['dir_samples'] + '/epoch_{:03d}'.format(epoch) #  + str(epoch)
    os.makedirs(save_dir, exist_ok=True)

    # if loss_meter.avg < best_loss or epoch % 10 == 0 or
    # 		epoch > 100 or epoch < 20:
    if True:
        print('\nSaving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        torch.save(state, save_dir + '/model.pth.tar')
        best_loss = loss_meter.avg

    # import ipdb; ipdb.set_trace()
    sample_fields = ['num_samples', 'in_channels', 'resize_hw']
    images, latent_z = sample(net, device=device, **filter_args( args, fields=sample_fields ) )

    # plot x and z
    num_samples = args['num_samples']
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    z_concat = torchvision.utils.make_grid(latent_z, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, save_dir+'/x.png')
    torchvision.utils.save_image(z_concat, save_dir+'/z.png')

    # with open(, 'wb') as z_serialize:
    # 	pickle.dump(latent_z, z_serialize)
    torch.save(latent_z, f = save_dir+'/z.pkl')

    # dict keys as returned by "train"
    train_loss = args['train_loss']
    train_bpd = args['train_bpd']
    report = [epoch, loss_meter.avg, bpd_meter.avg] + [train_loss, train_bpd]

    dir_samples = args['dir_samples']
    with open('{}/log'.format(dir_samples), 'a') as l:
        report = ", ".join([str(m) for m in report])
        report += "\n"
        print("\nWriting to disk:\n" + report + "At {}".format(dir_samples))
        l.write(report)


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

def find_last_model_relpath(fp):
    dirs_l = os.listdir(fp)
    dirs_e = [d for d in dirs_l if d.startswith('epoch_') 
                                     and d[-3:].isdigit()]
    dirs_e.sort()
    last_epoch = dirs_e[-1]
    print('Last model it.: ' + last_epoch)
    return fp + '/' + last_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')
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
    dataset_ = 'MNIST' 
    # dataset_ = 'celeba'
    # 2. Architecture
    net_ = 'densenet'  # 2.
    # 3. Samples dir_
    dir_ = '/1_' + net_[:3] +'_'+dataset_ if dataset_ == 'celeba' else '/res_3-8-32' # 3.
    dir_ = '/dense_test6'
    # only multi-gpu if celeba.resnet.
    # 4. GPUs
    gpus_ = '[0, 1]' if net_ == 'resnet' and dataset_=='celeba'  else '[0]' # 4.
    # 5. resume training?
    resume_ = True # 5.
    # 6. resize 

    if dataset_ == 'MNIST':
        in_channels_= 1
    elif dataset_ == 'celeba':
        in_channels_= 3
        resize_hw_ = '(64, 64)' # 6.

    if resume_:
        dir_model_ = find_last_model_relpath('data' + dir_)
    if 'resize_hw_' in dir():
        # TODO
        parser.add_argument('--resize_hw', default=resize_hw_, type=eval)

    parser.add_argument('--resume', '-r', action='store_true', default=resume_, help='Resume from checkpoint')
    parser.add_argument('--gpu_ids', default=gpus_, type=eval, help='IDs of GPUs to use')
    parser.add_argument('--net_type', default=net_, help='CNN architecture (resnet or densenet)')
    parser.add_argument('--dir_samples', default="data" + dir_ , help="Directory for storing generated samples")

    # dataset
    parser.add_argument('--dataset', '-ds', default=dataset_, type=str, help="MNIST or CIFAR-10")
    
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

        elif dataset_.upper() == 'MNIST': # data/dense_test6
            batch_size_ = 512
            mid_channels_ = 120
            num_levels_ = 10
            num_samples_ = 121
            num_scales_ = 3

    elif net_ == 'resnet':
        if dataset_.lower() == 'celeba': # CelebA 
            batch_size_ = 8 if len(gpus_) > 3 else 1
            mid_channels_ = 32
            num_levels_ = 8
            num_samples_ = 8
            num_scales_ = 2
        elif dataset_.upper() == 'MNIST': # data/dense_test6
            batch_size_ = 218
            mid_channels_ = 32
            num_levels_ = 8
            num_samples_ = 121
            num_scales_ = 3
            # time on TitanX-Pascal (batch_size == 8):
            # training : 8:07:44 / epoch
            # testing :  16:55   / epoch
            # time on RTX2080Ti (batch_size == 4):

    parser.add_argument('--batch_size', default=batch_size_, type=int, help='Batch size')
    parser.add_argument('--mid_channels', default=mid_channels_, type=int, help='N of feature maps for first resnet layer')
    parser.add_argument('--num_levels', default=num_levels_, type=int, help='N of residual blocks in resnet, or N of dense layers in densenet (depth)')
    parser.add_argument('--dir_model', default=dir_model_, help="Directory for storing generated samples")
    parser.add_argument('--num_samples', default=num_samples_, type=int, help='Number of samples at test time')
    parser.add_argument('--num_scales', default=num_scales_, type=int, help='Real NVP multi-scale arch. recursions')

    
    best_loss = 5e5

    main(parser.parse_args())
