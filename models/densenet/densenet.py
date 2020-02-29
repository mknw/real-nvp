"""
stealed from (add source)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


UNIT_TESTING = False


class Conv2dReLU(nn.Module):
	def __init__(
			self, in_c, out_c, kernel_size=3, stride=1, padding=0,
			bias=True):
		super().__init__()

		self.nn = nn.Conv2d(in_c, out_c, kernel_size, padding=padding)

	def forward(self, x):
		h = self.nn(x)

		y = F.relu(h)

		return y


class DenseLayer(nn.Module):
	def __init__(self, in_c, growth, Conv2dAct):
		super().__init__()

		conv1x1 = Conv2dAct(
				in_c, in_c, kernel_size=1, stride=1,
				padding=0, bias=True)

		self.nn = torch.nn.Sequential(
			conv1x1,
			Conv2dAct(
				in_c, growth, kernel_size=3, stride=1,
				padding=1, bias=True),
			)

	def forward(self, x):
		h = self.nn(x)
		#                       \__/
		#                       (o-)   # your life is ending.
		#                      //||\\
		h = torch.cat([x, h], dim=1) # found the BUG!!!
		return h


class DenseBlock(nn.Module):
	def __init__(
			self, in_c, out_c, kernel, Conv2dAct=Conv2dReLU, depth=8):
		# TODO: remove kernel in this module if it not needed.
		super().__init__()
		depth = depth

		future_growth = out_c - in_c

		layers = []

		for d in range(depth):
			# it 1:
			# growth = 0 / 8
			growth = future_growth // (depth - d)

			layers.append(DenseLayer(in_c, growth, Conv2dAct))
			in_c += growth
			future_growth -= growth

		self.nn = torch.nn.Sequential(*layers)

	def forward(self, x):
		return self.nn(x)


class DenseNet(nn.Module):
	def __init__(
			self, in_c, mid_c, out_c, kernel_size=3):
		'''
		DenseNet class for generative flow.
		in_c: input channels (1 for grayscale imgs, 3 for rgb)
		mid_c: amount of feature maps used in the network (512)
		out_c: output space (z, equals in_c * 2 for Real NVP)
		'''
		super(DenseNet, self).__init__()

		Conv2dAct = Conv2dReLU

		layers = [
			DenseBlock(
				in_c=in_c, # 1
				out_c=mid_c + in_c, # 512 + 1
				kernel=kernel_size,
				Conv2dAct=Conv2dAct,
				depth=8)]

		layers += [
			torch.nn.Conv2d(mid_c + in_c, out_c, kernel_size, padding=1)
			]

		self.nn = torch.nn.Sequential(*layers)

		# Set parameters of last conv-layer to zero.
		if not UNIT_TESTING:
			self.nn[-1].weight.data.zero_()
			self.nn[-1].bias.data.zero_()

	def forward(self, x):
		return self.nn(x)
