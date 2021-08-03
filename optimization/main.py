import click 

import torch as th 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
 
from libraries.strategies import * 
from libraries.log import logger 

from datalib.data_holder import DATAHOLDER 
from datalib.data_loader import DATALOADER 

from modelization.damsm import *
from modelization.generator import GENERATOR 
from modelization.discriminator import DISCRIMINATOR 

from os import path, mkdir 

def train_0(storage, nb_epochs, bt_size, pretrained_model, common_space_dim, snapshot_interval, dump_path):
	device = th.device( 'cuda:0' if th.cuda.is_available() else 'cpu' )
	
	source = DATAHOLDER(path_to_storage=storage, max_len=18, neutral='<###>')
	loader = DATALOADER(dataset=source, shuffle=True, batch_size=bt_size)
	
	network = DAMSM(vocab_size=len(source.vocab_mapper), common_space_dim=common_space_dim)
	network.to(device)
	
	solver = optim.Adam(network.parameters(), lr=0.002, betas=(0.5, 0.999))
	criterion = nn.CrossEntropyLoss().to(device)

	total_images = len(source)
	for epoch_counter in range(nb_epochs):
		nb_images = 0
		for index, (_, _, images, captions, lengths) in enumerate(loader.loader):
			batch_size = images.size(0)
			nb_images = nb_images + batch_size 

			images = images.to(device)
			captions = captions.to(device)

			labels = th.arange(len(images)).to(device)
			response = network(images, captions, lengths)	
			
			words, sentence, local_features, global_features = response 
			wq_prob, qw_prob = local_match_probabilities(words, local_features)
			sq_prob, qs_prob = global_match_probabilities(sentence, global_features)

			loss_w1 = criterion(wq_prob, labels) 
			loss_w2 = criterion(qw_prob, labels)
			loss_s1 = criterion(sq_prob, labels)
			loss_s2 = criterion(qs_prob, labels)

			loss_sw = loss_w1 + loss_w2 + loss_s1 + loss_s2

			solver.zero_grad()
			loss_sw.backward()
			solver.step()

			message = (nb_images, total_images, epoch_counter, nb_epochs, index, loss_sw.item())
			logger.debug('[%04d/%04d] | [%03d/%03d]:%05d >> Loss : %07.3f ' % message)

		if epoch_counter % snapshot_interval == 0:
			th.save(network, path.join(f'{dump_path}', f'damsm_{epoch_counter:03d}.th'))		
	
	th.save(network, path.join(f'{dump_path}' f'damsm_{epoch_counter:03d}.th'))


def train_1(storage, nb_epochs, bt_size, damsm_path, t_dim, c_dim, z_dim, common_space_dim, nb_gen_features, nb_dis_features, snapshot_interval, dump_path, images_store):
	device = th.device( 'cuda:0' if th.cuda.is_available() else 'cpu' )
	
	source = DATAHOLDER(path_to_storage=storage, max_len=18, neutral='<###>')
	loader = DATALOADER(dataset=source, shuffle=True, batch_size=bt_size)
	
	network = th.load(damsm_path, map_location=device)
	network.eval()
	
	for prm in network.parameters():
		prm.requires_grad = False 

	generator = GENERATOR(t_dim=t_dim, c_dim=c_dim, z_dim=z_dim, nb_gen_features=nb_gen_features) 
	discriminator_0 = DISCRIMINATOR(i_channels=3, o_channels=nb_dis_features, tdf=t_dim, img_size=64) 
	discriminator_1 = DISCRIMINATOR(i_channels=3, o_channels=nb_dis_features, tdf=t_dim, img_size=128)
	discriminator_2 = DISCRIMINATOR(i_channels=3, o_channels=nb_dis_features, tdf=t_dim, img_size=256)

	generator.to(device)
	discriminator_0.to(device)
	discriminator_1.to(device)
	discriminator_2.to(device)

	generator_solver = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
	discriminator_solver_0 = optim.Adam(discriminator_0.parameters(), lr=2e-4, betas=(0.5, 0.999))
	discriminator_solver_1 = optim.Adam(discriminator_1.parameters(), lr=2e-4, betas=(0.5, 0.999))
	discriminator_solver_2 = optim.Adam(discriminator_2.parameters(), lr=2e-4, betas=(0.5, 0.999))
	
	criterion_damsm = nn.CrossEntropyLoss().to(device)
	criterion_attngan = nn.BCELoss().to(device)

	LAMBDA = 5
	total_images = len(source)
	for epoch_counter in range(nb_epochs):
		nb_images = 0 
		for index, (r_image_064, r_image_128, r_image_256, captions, lengths) in enumerate(loader.loader):
			batch_size = r_image_256.size(0)
			nb_images = nb_images + batch_size 
			
			#-------------------#
			#move data to device#
			#-------------------#
			r_image_064 = r_image_064.to(device)
			r_image_128 = r_image_128.to(device)
			r_image_256 = r_image_256.to(device)
			
			#-------------------------#
			#DAMSM features extraction# 
			#-------------------------#
			captions = captions.to(device)
			real_labels = th.ones(batch_size).to(device)
			fake_labels = th.zeros(batch_size).to(device)
			damsm_labels = th.arange(batch_size).to(device)
			
			response = network.encode_seq(captions, lengths)	
			words, sentence = list(map(lambda M: M.detach(), response)) 
			
			#-----------------------#
			#TRAIN GENERATOR NETWORK#
			#-----------------------#
			noise = th.randn(batch_size, z_dim)
			f_image_064, f_image_128, f_image_256, mu, logvar = generator(noise, sentence, words)
			
			F00, F01 = discriminator_0.compute(f_image_064, sentence)
			F10, F11 = discriminator_1.compute(f_image_128, sentence) 
			F20, F21 = discriminator_2.compute(f_image_256, sentence)

			# compute multi generators loss 
			LF00 = criterion_attngan(F00, real_labels)  # uloss image_064
			LF01 = criterion_attngan(F01, real_labels)  # closs image_064
			
			LF10 = criterion_attngan(F10, real_labels)  # uloss image_128
			LF11 = criterion_attngan(F11, real_labels)  # closs image_128
			
			LF20 = criterion_attngan(F20, real_labels)  # uloss image_256
			LF21 = criterion_attngan(F21, real_labels)  # closs image_256

			G064 = LF00 + LF01  # Total loss for generator image_064
			G128 = LF10 + LF11  # Total loss for generator image_128
			G256 = LF20 + LF21  # Total loss for generator image_256

			GTOT = G064 + G128 + G256 # Total loss for all generators 
			
			# compute damsm loss 
			response = network.encode_img(f_image_256)	
			local_features, global_features = list(map(lambda M: M.detach(), response)) 
			
			wq_prob, qw_prob = local_match_probabilities(words, local_features)
			sq_prob, qs_prob = global_match_probabilities(sentence, global_features)

			loss_w1 = criterion_damsm(wq_prob, damsm_labels) 
			loss_w2 = criterion_damsm(qw_prob, damsm_labels)
			loss_s1 = criterion_damsm(sq_prob, damsm_labels)
			loss_s2 = criterion_damsm(qs_prob, damsm_labels)

			loss_damsm = loss_w1 + loss_w2 + loss_s1 + loss_s2

			# compute KL_Loss
			KL = 0.5 * th.sum(th.exp(logvar) + mu ** 2 - logvar - 1)

			# add all losses 
			GTOT = GTOT + LAMBDA * loss_damsm + KL 

			generator_solver.zero_grad()
			GTOT.backward()
			generator_solver.step()

			#---------------------------#
			#TRAIN DISCRIMINATOR NETWORK#
			#---------------------------#
			R00, R01 = discriminator_0.compute(r_image_064, sentence)
			R10, R11 = discriminator_1.compute(r_image_128, sentence) 
			R20, R21 = discriminator_2.compute(r_image_256, sentence)			

			F00, F01 = discriminator_0.compute(f_image_064.detach(), sentence)
			F10, F11 = discriminator_1.compute(f_image_128.detach(), sentence) 
			F20, F21 = discriminator_2.compute(f_image_256.detach(), sentence)
			
			LR00 = criterion_attngan(R00, real_labels)  # uloss image_064
			LR01 = criterion_attngan(R01, real_labels)  # closs image_064
			
			LR10 = criterion_attngan(R10, real_labels)  # uloss image_128
			LR11 = criterion_attngan(R11, real_labels)  # closs image_128
			
			LR20 = criterion_attngan(R20, real_labels)  # uloss image_256
			LR21 = criterion_attngan(R21, real_labels)  # closs image_256
			
			LF00 = criterion_attngan(F00, fake_labels)  # uloss image_064
			LF01 = criterion_attngan(F01, fake_labels)  # closs image_064
			
			LF10 = criterion_attngan(F10, fake_labels)  # uloss image_128
			LF11 = criterion_attngan(F11, fake_labels)  # closs image_128
			
			LF20 = criterion_attngan(F20, fake_labels)  # uloss image_256
			LF21 = criterion_attngan(F21, fake_labels)  # closs image_256
			
			D064 = (LR00 + LR01 + LF00 + LF01) / 2  # Total loss discriminator image_064  
			D128 = (LR10 + LR11 + LF10 + LF11) / 2  # Total loss discriminator image_128
			D256 = (LR20 + LR21 + LF20 + LF21) / 2  # Total loss discriminator image_256 
			
			discriminator_solver_0.zero_grad()
			D064.backward()
			discriminator_solver_0.step()
			
			discriminator_solver_1.zero_grad()			
			D128.backward()
			discriminator_solver_1.step()
			
			discriminator_solver_2.zero_grad()						
			D256.backward()
			discriminator_solver_2.step()			

			message = (nb_images, total_images, epoch_counter, nb_epochs, index, GTOT.item(), D064.item(), D128.item(), D256.item())
			logger.debug('[%04d/%04d] | [%03d/%03d]:%05d >> GLoss : %07.3f | D064 : %07.3f | D128 : %07.3f | D256 : %07.3f' % message)

			if index % snapshot_interval == 0:	
				descriptions = [ source.map_index2caption(seq) for seq in captions]
				output = snapshot(r_image_256.cpu(), f_image_256.cpu(), descriptions, f'output epoch {epoch_counter:03d}', mean=[0.5], std=[0.5])
				cv2.imwrite(path.join(images_store, f'###_{epoch_counter:03d}_{index:05d}.jpg'), output)

		if epoch_counter % 100 == 0:
			th.save(generators, path.join(f'{dump_path}', f'generators_{epoch_counter:03d}.th'))		
			th.save(discriminator_0, path.join(f'{dump_path}', f'D064_{epoch_counter:03d}.th'))		
			th.save(discriminator_1, path.join(f'{dump_path}', f'D128_{epoch_counter:03d}.th'))		
			th.save(discriminator_2, path.join(f'{dump_path}', f'D256_{epoch_counter:03d}.th'))		
					

	th.save(generators, path.join(f'{dump_path}', f'generators_{epoch_counter:03d}.th'))		
	th.save(discriminator_0, path.join(f'{dump_path}', f'D064_{epoch_counter:03d}.th'))		
	th.save(discriminator_1, path.join(f'{dump_path}', f'D128_{epoch_counter:03d}.th'))		
	th.save(discriminator_2, path.join(f'{dump_path}', f'D256_{epoch_counter:03d}.th'))		
		

@click.command()
@click.option('--storage', help='path to dataset: [CUB]')
@click.option('--nb_epochs', help='number of epochs', type=int)
@click.option('--bt_size', help='batch size', type=int)
@click.option('--pretrained_model', help='path to pretrained damsm model', default='')
@click.option('--common_space_dim', help='', default=256, type=int)
@click.option('--snapshot_interval', help='interval of saving damsm models', type=int, default=100)
@click.option('--dump_path', help='path to models store', type=click.Path(False), default='dump')
@click.pass_context
def damsm_training(ctx, storage, nb_epochs, bt_size, pretrained_model, common_space_dim, snapshot_interval, dump_path):
	if not path.isdir(dump_path):
		mkdir(dump_path)
	train_0(storage, nb_epochs, bt_size, pretrained_model, common_space_dim, snapshot_interval, dump_path)


@click.command()
@click.option('--storage', help='path to dataset: [CUB]')
@click.option('--nb_epochs', help='number of epochs', type=int)
@click.option('--bt_size', help='batch size', type=int)
@click.option('--damsm_path', help='path to pretrained damsm', type=click.Path(True))
@click.option('--t_dim', help='sentence features dim', type=int, default=256)
@click.option('--c_dim', help='condition augmentation dim', type=int, default=64)
@click.option('--z_dim', help='noise vector dimension', type=int, default=100)
@click.option('--common_space_dim', help='', default=256, type=int)
@click.option('--nb_gen_features', help='number extracted features for generator', type=int, default=32)
@click.option('--nb_dis_features', help='number extracted features for discriminator', type=int, default=64)
@click.option('--snapshot_interval', help='interval of saving images', type=int, default=100)
@click.option('--dump_path', help='path to models store', type=click.Path(False), default='dump')
@click.option('--images_store', help='path to generated images store', type=click.Path(False), default='images_store')
@click.pass_context
def attngan_training(ctx, storage, nb_epochs, bt_size, damsm_path, t_dim, c_dim, z_dim, common_space_dim, nb_gen_features, nb_dis_features, snapshot_interval, dump_path, images_store):
	if not path.isdir(dump_path):
		mkdir(dump_path)
	if not path.isdir(images_store):
		mkdir(images_store)	
	train_1(storage, nb_epochs, bt_size, damsm_path, t_dim, c_dim, z_dim, common_space_dim, nb_gen_features, nb_dis_features ,snapshot_interval, dump_path, images_store)


@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='debug Flag', default=False)
@click.pass_context
def main_command(ctx, debug):
	if not ctx.invoked_subcommand:
		logger.debug('main command')

main_command.add_command(damsm_training)
main_command.add_command(attngan_training)

if __name__ == '__main__':
	main_command(obj={})
