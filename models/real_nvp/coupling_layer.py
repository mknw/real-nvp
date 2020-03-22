import torch
import torch.nn as nn

from enum import IntEnum
from util import checkerboard_mask


class MaskType(IntEnum):
	CHECKERBOARD = 0
	CHANNEL_WISE = 1


class CouplingLayer(nn.Module):
	"""Coupling layer in RealNVP.

	Args:
		in_c (int): Number of channels in the input.
		mid_c (int): Number of channels in the `s` and `t` network.
		num_blocks (int): Number of residual blocks in the `s` and `t` network.
		mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
		reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
		net_type (str): densenet or resnet
	"""
	def __init__(self, in_c, mid_c, num_levels, mask_type, reverse_mask,
							net_type=None, **kwargs):
		super(CouplingLayer, self).__init__()

		# Save mask info
		self.mask_type = mask_type
		self.reverse_mask = reverse_mask

		# Build scale and translate network
		if self.mask_type == MaskType.CHANNEL_WISE:
			in_c //= 2

		if net_type == "resnet":
			from models.resnet import ResNet
			self.st_net = ResNet(in_c, mid_c, 2 * in_c,
							 num_blocks=num_levels, kernel_size=3, padding=1,
							 double_after_norm=(self.mask_type == MaskType.CHECKERBOARD))
		elif net_type == "densenet":
			from models.densenet import DenseNet
			self.st_net = DenseNet(in_c=in_c, mid_c=mid_c, out_c=2*in_c, 
					          depth=num_levels, double=(self.mask_type == MaskType.CHECKERBOARD))
		else:
			raise NotImplementedError('net type should be `resnet` or `densenet`')

		# Learnable scale for s
		self.rescale = nn.utils.weight_norm(Rescale(in_c))

	def forward(self, x, sldj=None, reverse=True):
		if self.mask_type == MaskType.CHECKERBOARD:
			# Checkerboard mask
			b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
			x_b = x * b
			st = self.st_net(x_b)
			s, t = st.chunk(2, dim=1)
			s = self.rescale(torch.tanh(s))
			s = s * (1 - b)
			t = t * (1 - b)

			# Scale and translate
			if reverse:
				inv_exp_s = s.mul(-1).exp()
				if torch.isnan(inv_exp_s).any():
					raise RuntimeError('Scale factor has NaN entries')
				x = x * inv_exp_s - t
			else:
				exp_s = s.exp()
				if torch.isnan(exp_s).any():
					import ipdb; ipdb.set_trace()
					'''might have to normalize nn input in range [-1, 1]'''
					raise RuntimeError('Scale factor has NaN entries')
				x = (x + t) * exp_s

				# Add log-determinant of the Jacobian
				sldj += s.view(s.size(0), -1).sum(-1)
		else:
			# Channel-wise mask
			if self.reverse_mask:
				x_id, x_change = x.chunk(2, dim=1)
			else:
				x_change, x_id = x.chunk(2, dim=1)

			st = self.st_net(x_id)
			s, t = st.chunk(2, dim=1)
			s = self.rescale(torch.tanh(s)) # scale s by tensor

			# Scale and translate
			if reverse:
				inv_exp_s = s.mul(-1).exp()
				if torch.isnan(inv_exp_s).any():
					raise RuntimeError('Scale factor has NaN entries')
				x_change = x_change * inv_exp_s - t
			else:
				exp_s = s.exp()
				if torch.isnan(exp_s).any():
					raise RuntimeError('Scale factor has NaN entries')
				x_change = (x_change + t) * exp_s

				# Add log-determinant of the Jacobian
				sldj += s.view(s.size(0), -1).sum(-1)

			if self.reverse_mask:
				x = torch.cat((x_id, x_change), dim=1)
			else:
				x = torch.cat((x_change, x_id), dim=1)

		return x, sldj


class Rescale(nn.Module):
	"""Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
	with `torch.nn.utils.weight_norm`.

	Args:
		num_channels (int): Number of channels in the input.
	"""
	def __init__(self, num_channels):
		super(Rescale, self).__init__()
		self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

	def forward(self, x):
		x = self.weight * x
		return x
