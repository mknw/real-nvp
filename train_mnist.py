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
		# net = RealNVP(num_scales=2, in_channels=1, mid_channels=64, num_blocks=8, **args.__dict__)
		net = RealNVP( **filter_args(args.__dict__) )

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
		test(epoch, net, testloader, device, loss_fn, args.num_samples, args.dir_samples, **train_stats)


def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
	print('\nEpoch: %d' % epoch)
	net.train()
	loss_meter = util.AverageMeter()
	bpd_meter = util.AverageMeter()
	with tqdm(total=len(trainloader.dataset)) as progress_bar:
		for x, _ in trainloader:
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

def sample(net, batch_size, device):
	"""Sample from RealNVP model.

	Args:
		net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
		batch_size (int): Number of samples to generate.
		device (torch.device): Device to use.
	"""
	# side_size = 28 if net.in_channels == 1 else 32 # debatable.. but ok for now.
	z = torch.randn((batch_size, 1, 28, 28), dtype=torch.float32, device=device) #changed 3 -> 1
	x, _ = net(z, reverse=True)
	x = torch.sigmoid(x)

	return x, z


def test(epoch, net, testloader, device, loss_fn, num_samples, dir_samples, **args):
	global best_loss
	net.eval()
	loss_meter = util.AverageMeter()
	bpd_meter = util.AverageMeter()
	with torch.no_grad():
		with tqdm(total=len(testloader.dataset)) as progress_bar:
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
				#debugopt:
				
	# Save checkpoint
	save_dir = dir_samples+"/epoch_"+str(epoch)
	os.makedirs(save_dir, exist_ok=True)

	if loss_meter.avg < best_loss or epoch % 10 == 0:
	# if epoch > 40:
		print('\nSaving...')
		state = {
			'net': net.state_dict(),
			'test_loss': loss_meter.avg,
			'epoch': epoch,
		}
		torch.save(state, dir_samples+ '/model.pth.tar') # TODO: change dir_samples to model_dir
		best_loss = loss_meter.avg

	# Save samples and data
	images, latent_z = sample(net, num_samples, device)

	# plot x and z
	images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
	z_concat = torchvision.utils.make_grid(latent_z, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
	torchvision.utils.save_image(images_concat, save_dir+'/x.png')
	torchvision.utils.save_image(z_concat, save_dir+'/z.png')

	import pickle
	with open(save_dir+'/z.pkl', 'wb') as z_serialize:
		pickle.dump(latent_z, z_serialize)

	# dict keys as returned by "train"
	train_loss = args['train_loss']
	train_bpd = args['train_bpd']
	report = [epoch, loss_meter.avg, bpd_meter.avg] + [train_loss, train_bpd]

	with open('{}/log'.format(dir_samples), 'a') as l:
		report = ", ".join([str(m) for m in report])
		report += "\n"
		print("\nWriting to disk:\n" + report + "At {}".format(dir_samples))
		l.write(report)


def filter_args(arg_dict, arch_fields=None):
	"""only pass to network architecture relevant fields."""
	if not arch_fields:
		# if arg_dict['net_type'] == 'resnet':
		arch_fields = ['net_type', 'num_scales', 'in_channels', 'mid_channels', 'num_levels']
		# elif arch['net_type'] == 'densenet':
		# 	arch_fields = ['net_type', 'num_scales', 'in_channels', 'mid_channels', 'depth']
	return {k:arg_dict[k] for k in arch_fields if k in arg_dict}


class GaussianNoise(object):

	def __init__(self, mean=0., std=.1, restrict_range=True):
		self.std = std
		self.mean = mean
		self.restrict_range = restrict_range

	def __call__(self, tensor):
		tensor += torch.randn(tensor.size()) * self.std + self.mean
		if self.restrict_range:
			return tensor.clamp(1e-5, 1)
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

	# test_batch_size = 1000 # ?
	parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
	parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
	parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
	''' resnet dir_sample '''
	parser.add_argument('--dir_samples', default="data/dense_test4", help="Directory for storing generated samples")
	''' densenet dir_sample '''
	# parser.add_argument('--dir_samples', default="data/2dense_3-8-128", help="Directory for storing generated samples")

	# parser.add_argument('--dir_model', default="data/dense_3-16-128/epoch_50", help="Directory for storing generated samples")
	# parser.add_argument('--dir_model', default="data/dense_test4/epoch___", help="Directory for storing generated samples")
	parser.add_argument('--dir_model', default="data/dense_test/epoch_80", help="Directory for storing generated samples")
	parser.add_argument('--resume', '-r', action='store_true', default=True, help='Resume from checkpoint')

	parser.add_argument('--dataset', '-ds', default="MNIST", type=str, help="MNIST or CIFAR-10")
	parser.add_argument('--in_channels', default=1, type=int, help='dimensionality along Channels')
	# Hyperparameters
	# training
	# parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
	parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
	parser.add_argument('--num_samples', default=121, type=int, help='Number of samples at test time')

	parser.add_argument('--num_epochs', default=200, type=int, help='Number of epochs to train')
	parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate') # changed from 1e-3 for MNIST
	parser.add_argument('--weight_decay', default=5e-5, type=float,
											help='L2 regularization (only applied to the weight norm scale factors)')
	parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
	# Test

	# General architecture parameters
	net = 'densenet'
	parser.add_argument('--net_type', default=net, help='CNN architecture (resnet or densenet)')

	if net == 'densenet':
		parser.add_argument('--num_scales', default=3, type=int, help='Real NVP multi-scale arch. recursions')
		parser.add_argument('--mid_channels', default=256, type=int, help='N of feature maps for first resnet layer')
		parser.add_argument('--num_levels', default=16, type=int, help='N of residual blocks in resnet, or N of dense layers in densenet (depth)')
	elif net == 'resnet':
		parser.add_argument('--num_scales', default=3, type=int, help='Real NVP multi-scale arch. recursions')
		parser.add_argument('--mid_channels', default=32, type=int, help='N of feature maps for first resnet layer')
		parser.add_argument('--num_levels', default=8, type=int, help='N of residual blocks in resnet, or N of dense layers in densenet (depth)')

	
	best_loss = 1500

	main(parser.parse_args())
