"""
stealed from (add source)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from util import WNConv2dReLU


UNIT_TESTING = True


class Conv2dReLU(nn.Module):
	def __init__(
			self, in_c, out_c, kernel_size=3, stride=1, padding=0,
			bias=True):
		super().__init__()

		self.nn = nn.Conv2d(in_c, out_c, kernel_size, padding=padding)

	def forward(self, x):
		h = self.nn(x)
		y = F.relu(h)
		if torch.isnan(y).any():
			import ipdb; ipdb.set_trace()
		return y


class DenseLayer(nn.Module):
	def __init__(self, in_c, growth, Conv2dAct):
		super().__init__()

		# if in_c > 1:
		# 	bn = in_c // 2
		# else: bn = in_c

		# '''this only make sense if out_c of 1x1 == 4*k'''
		# conv1x1 = Conv2dAct( # change made: in_c -> in_c // 2
		# 		in_c, in_c, kernel_size=1, # stride=1, # since it's always 1.
		# 		padding=0, bias=False)

		# self.nn = torch.nn.Sequential(
		# 	conv1x1,
		# 	Conv2dAct( # change made: in_c -> in_c // 2
		# 		in_c, growth, kernel_size=3, # stride=1,
		# 		padding=1, bias=True),
		# 	)
		# in_c, growth, kernel_size=3, # stride=1,
		'''alternative'''
		self.nn = Conv2dAct(
				in_c, growth, kernel_size=3, # stride=1,
				padding=1, bias=True)

	def forward(self, x):
		h = self.nn(x)
		h = torch.cat([x, h], dim=1) 
		return h


class DenseBlock(nn.Module):
	def __init__(
			self, in_c, out_c, kernel, Conv2dAct, depth=8): # Conv2ReLU
		# TODO: remove kernel in this module if it not needed.
		# TODO: give `depth` argument to training script.
		super().__init__()
		# depth = depth

		future_growth = out_c - in_c
		# k = future_growth // depth # for conv1x1 dimensionality reduction.

		layers = []

		for d in range(depth):
			growth = future_growth // (depth - d)

			layers.append(DenseLayer(in_c, growth, Conv2dAct))
			
			in_c += growth
			future_growth -= growth

		if future_growth != 0:
			print("future growth not zero: " + str(future_growth))
			import ipdb; ipdb.set_trace()


		self.nn = torch.nn.Sequential(*layers)

	def forward(self, x):
		return self.nn(x)


class DenseNet(nn.Module):
	def __init__(
			self, in_c, mid_c, out_c, depth=8, kernel_size=3, double=None):
		'''
		DenseNet class for generative flow.
		in_c: input channels (1 for grayscale imgs, 3 for rgb)
		mid_c: amount of feature maps used in the network (512)
		out_c: output space (z, equals in_c * 2 for Real NVP)
		'''
		super(DenseNet, self).__init__()
		self.double_after_norm = double

		Conv2dAct = WNConv2dReLU

		self.in_norm = nn.BatchNorm2d(in_c)
		in_c *= 2 # for both s and t

		layers = [
				DenseBlock( in_c=in_c, # 1
					out_c=mid_c + in_c, # 512 + 1
					kernel=kernel_size,
					Conv2dAct=Conv2dAct,
					depth=depth)
				]

		layers += [
			torch.nn.Conv2d(mid_c + in_c, out_c, kernel_size, padding=1)
			]
		self.nn = torch.nn.Sequential(*layers)

		self.out_norm = nn.BatchNorm2d(out_c)

		# Set parameters of last conv-layer to zero.
		if not UNIT_TESTING:
			self.nn[-1].weight.data.zero_()
			self.nn[-1].bias.data.zero_()

	def forward(self, x):
		# import ipdb; ipdb.set_trace()
		x = self.in_norm(x)
		if self.double_after_norm:
			x *= 2
		x = torch.cat((x, -x), dim=1)
		x = self.nn(x)
		x = self.out_norm(x)
		return x
