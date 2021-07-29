import torch as th 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 

class CA_BLOCK(nn.Module):
	def __init__(self, t_dim, c_dim):
		super(CA_BLOCK, self).__init__()
		self.lin0 = nn.Linear(t_dim, c_dim)  
		self.lin1 = nn.Linear(t_dim, c_dim)

	def forward(self, T):
		mu = self.lin0(T)
		lv = self.lin1(T)
		ca = mu + th.randn(mu.shape) * th.exp(lv / 2) 
		return ca, mu, lv 

class AT_BLOCK(nn.Module):
	def __init__(self, t_dim, h_dim):
		super(AT_BLOCK, self).__init__()
		self.head = nn.Conv2d(t_dim, h_dim, 1, 1, 0, bias=False)

	def forward(self, H, W):
		S = H.shape
		H = th.flatten(H, start_dim=2)           # BxDxH
		W = th.squeeze(self.head(W[..., None]))  # BxDxW
		B = th.softmax(th.einsum('ijk,ijm->ikm', H, W), dim=2) # BxHxW
		
		C = th.einsum('ijk,imk->ijm', B, W)      # BxHxW:BxDxW => BxHxD
		C = th.transpose(C, 1, 2)
		C = th.reshape(C, S)
		return C 

class UP_BLOCK(nn.Module):
	def __init__(self, i_channels, o_channels):
		super(UP_BLOCK, self).__init__()
		self.body = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(i_channels, o_channels, 3, 1, 1, bias=False),
			nn.BatchNorm2d(o_channels),
			nn.GELU()
		)

	def forward(self, X):
		return self.body(X)

class RS_BLOCK(nn.Module):
	def __init__(self, i_channels):
		super(RS_BLOCK, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(i_channels, i_channels * 2, 3, 1, 1, bias=False),
			nn.BatchNorm2d(i_channels * 2),
			nn.GELU(),
			nn.Conv2d(i_channels * 2, i_channels, 3, 1, 1, bias=False),
			nn.BatchNorm2d(i_channels),
		)

	def forward(self, X):
		return X + self.body(X)

class IM_BLOCK(nn.Module):
	def __init__(self, i_channels):
		super(IM_BLOCK, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(i_channels, 3, 1, 1, bias=False), 
			nn.Tanh()
		)

	def forward(self, X):
		return self.body(X)

class F0_BLOCK(nn.Module):
	def __init__(self, z_dim, c_dim, o_channels):
		super(F0_BLOCK, self).__init__()
		self.o_channels = o_channels
		self.head = nn.Sequential(
			nn.Linear(z_dim + c_dim, o_channels * 4 * 4, bias=False),
			nn.BatchNorm1d(o_channels * 4 * 4),
			nn.GELU()
		)
		self.body = nn.Sequential(*[
			UP_BLOCK(o_channels // 2 ** idx, o_channels // 2 ** (idx + 1))
			for idx in range(4)
		])

	def forward(self, Z, C):
		ZC = th.cat([Z, C], dim=1)
		X0 = self.head(ZC).view(-1, self.o_channels, 4, 4)
		return self.body(X0)  # decrease 16-times the channels

class FN_BLOCK(nn.Module):
	def __init__(self, t_dim, h_dim, nb_rs_BLOCK):
		super(FN_BLOCK, self).__init__()
		self.head = AT_BLOCK(t_dim=t_dim, h_dim=h_dim)
		self.body = nn.Sequential(*[
			RS_BLOCK(h_dim * 2)  # due to the concatenation of H_i and F_Attn(H_I, W_I)
			for _ in range(nb_rs_BLOCK)
		])
		self.term = UP_BLOCK(h_dim * 2, h_dim)

	def forward(self, H, W):
		A = self.head(H, W)
		HA = th.cat([H, A], dim=1)
		return self.term(self.body(HA))
		

class GENERATOR(nn.Module):
	def __init__(self, t_dim, c_dim, z_dim, nb_gen_features=32, nb_rs_BLOCK=2):
		super(GENERATOR, self).__init__()
		self.CA = CA_BLOCK(t_dim, c_dim)
		self.F0 = F0_BLOCK(z_dim, c_dim, nb_gen_features * 16)
		self.G0 = IM_BLOCK(nb_gen_features) 
		self.F1 = FN_BLOCK(t_dim, nb_gen_features, nb_rs_BLOCK)
		self.G1 = IM_BLOCK(nb_gen_features) 
		self.F2 = FN_BLOCK(t_dim, nb_gen_features, nb_rs_BLOCK)
		self.G2 = IM_BLOCK(nb_gen_features) 

	def forward(self, Z0, T0, W0):
		TT = th.transpose(T0, 0, 1)  # HxB => BxH 
		C0, MU, LV = self.CA(TT)

		X0 = self.F0(Z0, C0)
		X1 = self.F1(X0, W0)
		X2 = self.F2(X1, W0)

		I0 = self.G0(X0)
		I1 = self.G1(X1)
		I2 = self.G2(X2)

		return I0, I1, I2, MU, LV 

if __name__ == '__main__':
	G = GENERATOR(t_dim=256, c_dim=64, z_dim=100, nb_gen_features=32)
	print(G)
	
	Z = th.randn(4, 100)
	T = th.randn(256, 4)
	W = th.randn(4, 256, 18)
	R = G(Z, T, W)

