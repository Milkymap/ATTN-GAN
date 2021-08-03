import numpy as np 

import torch as th 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 

class DW_BLOCK(nn.Module):
	def __init__(self, i_channels, o_channels, regularize=True):
		super(DW_BLOCK, self).__init__()

		self.body = nn.Sequential(
			nn.Conv2d(i_channels, o_channels, 4, 2, 1, bias=False),
			nn.BatchNorm2d(o_channels) if regularize else nn.Identity(),
			nn.LeakyReLU(0.2, inplace=True)
		)

	def forward(self, X):
		return self.body(X)


class DISCRIMINATOR(nn.Module):
	def __init__(self, i_channels, o_channels, tdf, img_size):
		super(DISCRIMINATOR, self).__init__()

		self.nb_steps = int(np.log2(img_size)) - 2
		self.head = DW_BLOCK(i_channels, o_channels, False)
		self.body = nn.Sequential(*[
			DW_BLOCK(o_channels * 2 ** idx, o_channels * 2 ** (idx + 1))
			for idx in range(self.nb_steps - 1)
		])

		self.ccnn = nn.Sequential(
			nn.Conv2d(o_channels * 2 ** (self.nb_steps - 1) + tdf, o_channels * 2 ** (self.nb_steps - 1), 3, 1, 1, bias=False),
			nn.BatchNorm2d(o_channels * 2 ** (self.nb_steps - 1)),
			nn.LeakyReLU(0.2, inplace=True), 
			nn.Conv2d(o_channels * 2 ** (self.nb_steps - 1), 1, 4, 4, 1, bias=False),
			nn.Sigmoid()
		)
		self.ucnn = nn.Sequential(
			nn.Conv2d(o_channels * 2 ** (self.nb_steps - 1), 1, 4, 4, 1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, X0, T=None):
		X1 = self.body(self.head(X0))
		if T is not None:
			TT = th.transpose(T, 0, 1)
			_, _, m, n = X1.shape
			XT = th.cat( (X1, TT[:, :, None, None].repeat(1, 1, m, n)), dim=1 )
			X2 = self.ccnn(XT)
		else:
			X2 = self.ucnn(X1)
		return th.squeeze(X2) 

	def compute(self, X0, CA):
		return self(X0), self(X0, CA)

if __name__ == '__main__':
	pass 

	