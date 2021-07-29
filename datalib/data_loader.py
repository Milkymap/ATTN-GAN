import numpy as np 
import torch as th 

from .data_holder import DATAHOLDER 
from torch.utils.data import DataLoader 


from libraries.strategies import * 

class DATALOADER:
	def __init__(self, dataset, shuffle, batch_size, sampler=None):
		self.loader = DataLoader(
			dataset=dataset, 
			shuffle=shuffle, 
			batch_size=batch_size, 
			collate_fn=self.custom_collate_fn, 
			drop_last=True, 
			sampler=sampler 
		)

	def custom_collate_fn(self, data):
		images_064, images_128, images_256, captions, lengths = list(zip(*data))
		captions = th.stack(captions)
		
		images_064 = th.stack(images_064)
		images_128 = th.stack(images_128)
		images_256 = th.stack(images_256)
		
		sorted_index = np.argsort(lengths)[-1::-1].tolist()
		
		sorted_images_064 = images_064[sorted_index, ...]
		sorted_images_128 = images_128[sorted_index, ...]
		sorted_images_256 = images_256[sorted_index, ...]
		sorted_lengths = th.as_tensor(lengths)[sorted_index, ...]
		sorted_captions = captions[sorted_index, ...]
		return sorted_images_064, sorted_images_128, sorted_images_256, sorted_captions, sorted_lengths

if __name__ == '__main__':
	D = DATAHOLDER('storage')
	L = DATALOADER(D, True, 4)
	for idx, (img, cap, lng) in enumerate(L.loader): 
		fmg = th.sigmoid(th.randn(img.shape))
		txt = [ D.map_index2caption(seq) for seq in cap]
		output = snapshot(img, fmg, txt, f'output\nepoch {idx:03d}', mean=[0.5], std=[0.5])
		cv2.imshow('...', output)
		cv2.waitKey(0)
		if idx == 10:
			break 



