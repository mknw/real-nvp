#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

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
from random import randrange
import numpy as np
import matplotlib.pyplot as plt


def main(args):

	# debugging option:
	if args.net_type == 'densenet':
		# torch.backends.cudnn.enabled = False
		# torch.backends.cudnn.benchmark = True
		# torch.backends.cudnn.deterministic = True
		# args.num_samples = 42 # test this
		# print("cudnn backend disabled, sampling n=42")
		pass

	# device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
	device = torch.device("cuda:0" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
	print("evaluating on: %s" % device)

	#torchvision.transforms.Normalize((0.1307,), (0.3081,)) # mean, std, inplace=False.

	if args.dataset == 'MNIST':
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
	print()
	print(device)
	print()

	if str(device).startswith('cuda'):
		net = torch.nn.DataParallel(net, args.gpu_ids)
		cudnn.benchmark = args.benchmark

	# Load checkpoint.
	model_epoch = randrange(120, 250)
	model_epoch = 244
	print('selected model at {}th epoch'.format(model_epoch))
	args.dir_model = 'data/res_3-8-32/epoch_' + str(model_epoch)
		

	loss_fn = RealNVPLoss()
	# param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
	# optimizer = optim.Adam(param_groups, lr=args.lr, eps=1e-7)

	# train_stats = train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm)
	# test(epoch, net, testloader, device, loss_fn, args.num_samples, args.dir_samples)
	# check if stats have been already pickled to args.dir_model + '/z_mean_std.pkl'
	if os.path.isfile(args.dir_model + '/z_mean_std.pkl'):
		print('Found cached file, skipping computations of mean and std for each digit.')
		stats = torch.load(args.dir_model + '/z_mean_std.pkl')
	else:
		print('Loading model at ' + args.dir_model + '/model.pth.tar...')
		assert os.path.isdir(args.dir_model), 'Error: no checkpoint directory found!'
		checkpoint = torch.load(args.dir_model + '/model.pth.tar')
		'''
		# not sure how useful the next three lines are:::
		global best_loss
		best_loss = checkpoint['test_loss']
		# we start epoch after the saved one (avoids overwrites).
		start_epoch = checkpoint['epoch'] + 1
		'''
		try:
			net.load_state_dict(checkpoint['net'])
		except RuntimeError:
			raise ArchError("There's a problem importing the model, check parameters.")
		stats = track_distribution(net, testloader, device, loss_fn)

	torch.save(stats, args.dir_model + '/z_mean_std.pkl')
	scatter_alldigits(stats, args.dir_model + '/meanstds.png', model_epoch)
	scatter_eachdigit(stats, args.dir_model + '/dig_subplots.png', model_epoch)

def violin_eachdigit(stats, filename, m_epoch):
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
	          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
	          '#bcbd22', '#17becf']
	stds, means = [stats['std'], stats['mean']]
	# Steps to replicate for each subplot:
	fig = plt.figure(figsize=(8, 10))
	ax = fig.add_subplot()

	plt.title('z space stats for model at epoch {}'.format(m_epoch))
	plt.xlabel('mean')
	plt.ylabel('std')
	
	# axs.set_title('Stats for epoch {} overview'.format(m_epoch))
	dig = 0
	for col in range(ncols):
		for row in range(nrows):
			axs[row, col].scatter(means[dig], stds[dig], c=colors[dig],
					                  label=str(dig), alpha=.8, s=.8)
			# TODO: Plot average of all means and stds (per category).
			ctrd_x = np.mean(means[dig])
			ctrd_y = np.mean(stds[dig])
			axs[row, col].scatter(ctrd_x, ctrd_y, c=color_variant(colors[dig]))
			axs[row,col].set_xlabel('mean')
			axs[row,col].set_ylabel('std')
			axs[row,col].annotate('$\mu$: {:.2f}\n$\sigma^2$: {:.2f}'.format(ctrd_x, ctrd_y), (ctrd_x, ctrd_y))

			dig += 1

		if dig == 10: break

	fig.suptitle('Stats for epoch {} overview'.format(m_epoch))
	fig.tight_layout()
	fig.subplots_adjust(top=.70) # .88
	#for dig in range(10):
	#	plt.scatter(means[dig], stds[dig], c=colors[dig], label=str(dig), alpha=.8, s=.8)
	plt.savefig(filename, bbox_inches='tight')
	print('\nPlot saved to ' + filename)








def scatter_eachdigit(stats, filename, m_epoch):
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
	          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
	          '#bcbd22', '#17becf']
	stds, means = [stats['std'], stats['mean']]
	# Steps to replicate for each subplot:
	# plt.title('z space stats for model at epoch {}'.format(m_epoch))
	# plt.xlabel('mean')
	# plt.ylabel('std')
	nrows, ncols = (2, 5)
	fig, axs = plt.subplots(nrows, ncols, sharex='all', sharey='all', figsize=(10, 7))

	# axs.set_title('Stats for epoch {} overview'.format(m_epoch))
	dig = 0
	for col in range(ncols):
		for row in range(nrows):
			axs[row, col].scatter(means[dig], stds[dig], c=colors[dig],
					                  label=str(dig), alpha=.8, s=.8)
			# TODO: Plot average of all means and stds (per category).
			ctrd_x = np.mean(means[dig])
			ctrd_y = np.mean(stds[dig])
			axs[row, col].scatter(ctrd_x, ctrd_y, c=color_variant(colors[dig]))
			axs[row,col].set_xlabel('mean')
			axs[row,col].set_ylabel('std')
			axs[row,col].annotate('$\mu$: {:.2f}\n$\sigma^2$: {:.2f}'.format(ctrd_x, ctrd_y), (ctrd_x, ctrd_y))

			dig += 1

		if dig == 10: break

	fig.suptitle('Stats for epoch {} overview'.format(m_epoch))
	fig.tight_layout()
	fig.subplots_adjust(top=.88)
	#for dig in range(10):
	#	plt.scatter(means[dig], stds[dig], c=colors[dig], label=str(dig), alpha=.8, s=.8)
	plt.savefig(filename, bbox_inches='tight')
	print('\nPlot saved to ' + filename)


def color_variant(hex_color, brightness_offset=-50):
	""" takes a color like #87c95f and produces a lighter or darker variant """
	if len(hex_color) != 7:
		raise Exception("Passed %s into color_variant(), needs to be in #87c95f format." % hex_color)
	rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
	new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
	new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int] # make sure new values are between 0 and 255
	# hex() produces "0x88", we want just "88"
	hex_string = [hex(i)[2:] for i in new_rgb_int]
	hex_string = [val if len(val) == 2 else "0" + val for val in hex_string]
	return "#" + "".join(hex_string)


def scatter_alldigits(stats, filename, m_epoch):
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
	          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
	          '#bcbd22', '#17becf']
	fig, ax = plt.subplots()
	stds, means = [stats['std'], stats['mean']]
	plt.title('z space stats for model at epoch {}'.format(m_epoch))
	plt.xlabel('mean')
	plt.ylabel('std')

	for dig in range(10):
		ax.scatter(means[dig], stds[dig], c=colors[dig], label=str(dig), alpha=.8, s=.8)
	ax.legend()
	plt.savefig(filename, bbox_inches='tight')
	print('\nPlot saved to: ' + filename)



def track_distribution(net, loader, device, loss_fn, **kwargs):


	net.eval()
	loss_meter = util.AverageMeter()
	bpd_meter = util.AverageMeter()
	
	with tqdm(total=len(loader.dataset)) as progress:

		digits_stds = [np.array([]).reshape(0, 1) for i in range(10)]
		digits_means = [np.array([]).reshape(0, 1) for i in range(10)]

		for x, y in loader:
			x = x.to(device)
			z, sldj = net(x, reverse=False)
			mean_z = z.mean(dim=[1,2,3]) # `1` is channel.
			std_z = z.std(dim=[1,2,3])

			for digit in range(10):
				dig_std_z = std_z[ (y==digit).nonzero() ]
				dig_mean_z = mean_z[ (y==digit).nonzero() ]
				digits_stds[digit] = np.concatenate(( digits_stds[digit], dig_std_z.to('cpu').detach().numpy() ))
				digits_means[digit] = np.concatenate(( digits_means[digit], dig_mean_z.to('cpu').detach().numpy() ))

			loss = loss_fn(z, sldj)
			loss_meter.update(loss.item(), x.size(0))
			bpd_meter.update(util.bits_per_dim(x, loss_meter.avg), x.size(0))

			progress.set_postfix(loss=loss_meter.avg, 
					                 bpd=bpd_meter.avg)
			progress.update(x.size(0))
	# add loss.avg() and bpd.avg() somewhere in the plot. 
	
	return {'std': digits_stds, 'mean': digits_means}


def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
	print('\nEpoch: %d' % epoch)
	net.train()
	loss_meter = util.AverageMeter()
	bpd_meter = util.AverageMeter()
	with tqdm(total=len(trainloader.dataset)) as progress_bar:
		for x, _ in trainloader:
			x = x.to(device)
			optimizer.zero_grad()
			z, sldj = net(x, reverse=False)
			loss = loss_fn(z, sldj)
			loss_meter.update(loss.item(), x.size(0))
			# import ipdb; ipdb.set_trace()
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
				## HERE: TODO: do something with z and y
				loss = loss_fn(z, sldj)
				loss_meter.update(loss.item(), x.size(0))
				# bits per dimensions
				bpd_meter.update(util.bits_per_dim(x, loss_meter.avg), x.size(0))

				progress_bar.set_postfix(loss=loss_meter.avg,
																 bpd=bpd_meter.avg)
				progress_bar.update(x.size(0))

	# Save checkpoint
	save_dir = dir_samples+"/epoch_"+str(epoch)
	os.makedirs(save_dir, exist_ok=True)

	if loss_meter.avg < best_loss:
	# if epoch > 40:
		print('Saving...')
		state = {
			'net': net.state_dict(),
			'test_loss': loss_meter.avg,
			'epoch': epoch,
		}
		torch.save(state, save_dir + '/model.pth.tar')
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


class ArchError(Exception):
	def __init__(self, expression, message):
		self.expression = expression
		if not message:
			self.message = "State dictionary not matching your architecture. Check your params."
		else:
			self.message = message


def filter_args(arg_dict, desired_fields=None):
	"""only pass to network architecture relevant fields."""
	if not desired_fields:
		desired_fields = ['net_type', 'num_scales', 'in_channels', 'mid_channels', 'num_blocks']
	return {k:arg_dict[k] for k in desired_fields if k in arg_dict}


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

	# test_batch_size = 1000 # ?
	parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
	parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
	parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
	# dirs for save and load
	# parser.add_argument('--dir_samples', default="data/restest", help="Directory for storing generated samples")
	# parser.add_argument('--dir_model', default="data/res_____/epoch_199", help="Directory for storing generated samples")
	# parser.add_argument('--resume', '-r', action='store_true', default=False, help='Resume from checkpoint')
	parser.add_argument('--dataset', '-ds', default="MNIST", type=str, help="MNIST or CIFAR-10")
	# Hyperparameters
	# training
	parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
	parser.add_argument('--num_samples', default=121, type=int, help='Number of samples at test time')

	parser.add_argument('--num_epochs', default=1, type=int, help='Number of epochs to train')
	# parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate') # changed from 1e-3 for MNIST
	# parser.add_argument('--weight_decay', default=5e-5, type=float,
											# help='L2 regularization (only applied to the weight norm scale factors)')
	# parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
	# Test

	# General architecture parameters
	parser.add_argument('--net_type', default='resnet', help='CNN architecture (resnet or densenet)')
	parser.add_argument('--num_scales', default=3, type=int, help='Real NVP multi-scale arch. recursions')
	parser.add_argument('--in_channels', default=1, type=int, help='dimensionality along Channels')
	parser.add_argument('--mid_channels', default=32, type=int, help='N of feature maps for first resnet layer')

	# RESNET
	parser.add_argument('--num_blocks', default=8, type=int, help='N of residual blocks in resnet')


	
	# best_loss = 0

	main(parser.parse_args())
