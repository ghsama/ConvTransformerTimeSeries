from data_prep import *
from net import *
from evaluate import *
from torch import nn

import torch

import os
import argparse
import json

import datetime

def main_train(data_dir,dataset,meta_information_path,model_config_path,train_config,save_path):

	print('\n\n# Loading the configurations :')
	# Metainformation :
	with open(meta_information_path, 'r') as file:
		meta_information = json.load(file)

	# model config :
	with open(model_config_path, 'r') as file:
		model_config = json.load(file)

	print('# Used model_config :', model_config)
	
	# train config
	train_config_path = train_config
	with open(train_config_path, 'r') as file:
		train_config = json.load(file)
	
	print('# Used train_config :', train_config)
	
	# Checking if it is the default configuration
	try :
		if model_config['default'] == True :
			print('\n\t# WARNING : YOU ARE USING THE DEFAULT model_config.json GENERATED DURING DATA PREPARATION, YOU CAN MODIFY IT AT :',model_config_path)
	except :
		pass
	try :
		if model_config['default'] == True :
			print('\n\t# WARNING : YOU ARE USING THE DEFAULT train_config.json GENERATED DURING DATA PREPARATION, YOU CAN MODIFY IT AT :',train_config_path)
	except:
		pass

	# Saving path :
	os.makedirs(save_path+'/'+dataset+'/', exist_ok = True)
	save_path = save_path+'/'+dataset+'/'

	# Datasets Loading :

	batch_size = train_config['batch_size'] #32
	predict_batch_size = train_config['predict_batch_size']#31

	train_set = TrainDataset(data_dir, dataset)
	test_set = TestDataset(data_dir, dataset)

	# Datasets Sampling
	sampler = WeightedSampler(data_dir, dataset) # Use weighted sampler instead of random sampler
	train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0)
	test_loader = DataLoader(test_set, batch_size=predict_batch_size, sampler=RandomSampler(test_set), num_workers=0)

	# timeseries_list :
	train_timeseries_list = np.load(os.path.join(data_dir, f'train_timeseries_dict_{dataset}.npy'))
	test_timeseries_list = np.load(os.path.join(data_dir, f'test_timeseries_dict_{dataset}.npy'))
	
	# Device :
	device = 'cpu'
	if torch.cuda.is_available():
		torch.set_default_tensor_type(torch.cuda.FloatTensor)
		device = 'cuda'
	
	# Model Definition :
	windows_size = meta_information['window_size'] #args.window_size  # window learning size 
	number_of_vars = meta_information['number_of_vars']#4 # the covariates + the time serie to learn
	num_timeseries_kinds = meta_information['num_timeseries_kinds']#1 

	headers =  model_config['headers']#3
	depth = model_config['depth']#3
	kernel_size = model_config['kernel_size']#6

	# Model 
	model = ForcastConvTransformer(k = number_of_vars, headers=headers, depth=depth, seq_length=windows_size, kernel_size=kernel_size, num_tokens=num_timeseries_kinds)

	# Global window is the size of the 
	global_window_size = windows_size
	stride = 1

	# Global Parametrers
	windows_total = (global_window_size - (windows_size-stride))//stride # total number of windows train by global_size window

	# Learning params :

	# Saving parh
	PATH = save_path+'/ConvTransformer_'+dataset+str(datetime.datetime.now())+'.pth'

	created_model_config = dict()
	created_model_config['k'] = number_of_vars
	created_model_config['headers'] = headers
	created_model_config['depth'] = depth
	created_model_config['seq_length'] = windows_size
	created_model_config['kernel_size'] = kernel_size
	created_model_config['num_tokens'] = num_timeseries_kinds
	created_model_config['path'] = PATH

	with open(model_config_path, 'w') as file:
		json.dump(created_model_config, file)

	# Training

	num_epochs = train_config['num_epochs']
	lr_warmup = train_config['lr_warmup']
	learning_rate = train_config['learning_rate']


	# Optimizer
	opt = torch.optim.Adam(lr=learning_rate, params=model.parameters())

	# Training
	seen = 1
	best_test_loss = -1
	train_loss = []
	test_loss = []
	for epoch in range(num_epochs) :
		model.train(True)
		count = 0
		epoch_loss = 0
		print('Epoch',epoch)
		for i, (train_batch, tokens_idx, v_batch, labels_batch) in enumerate(train_loader):
			# Warm up learning rate :
			lr = max((learning_rate / lr_warmup) * seen, 1e-5)
			opt.lr = lr
			opt.zero_grad()

			# Transfer data to GPU
			train_batch = train_batch.to(torch.float32).to(device)  # not scaled
			labels_batch = labels_batch.to(torch.float32).to(device)  # not scaled
			v_batch = v_batch.to(torch.float32).to(device)  # scale
			tokens_idx = tokens_idx.to(device)

			# The loop on the windows on the global window
			st = 0
			nd = st + windows_size
			loss = 0
			for i in range(int(windows_total)):
				# the window extraction :
				x_train = train_batch[:,st:nd,:]
				v_train = v_batch
				batch_size = v_train.shape[0]
				
				# Model application :
				v_train_expanded = v_train.expand(windows_size,batch_size,2).transpose(1,0)
				estimation, mu, sigma = evaluate(model, x_train, tokens_idx, v_train_expanded, windows_size, 
												 sampling = False)

				# Loss calculation:
				loss += loss_fct(mu, sigma, labels_batch[:,st:nd], 'quantileloss_dist_normal')
				st+=1
				nd+=1
			# The mean loss
			loss = loss/int(windows_total)
			epoch_loss = loss
			loss.backward()
			
			# CLip gradient if > 1 .. 
			nn.utils.clip_grad_norm_(model.parameters(), 1)
			opt.step()
			
			seen += train_batch.size(0)*int(windows_total)
		if num_epochs > 100 :
			if (epoch % 100 == 0 and epoch !=0):
				print('Epoch number : ', epoch)
				print(f'-- "train" loss {epoch_loss:.4}')
		else :
			print(f'-- "train" loss {epoch_loss:.4}')
		
		# evaluation on test data 
		with torch.no_grad():
			model.train(False)
			test_loss = 0.0
			count = 0
			for i, (test_batch, tokens_tst_idx, v_tst, labels_tst) in enumerate(test_loader):
				 # Transfer data to GPU
				test_batch = test_batch.to(torch.float32).to(device)
				labels_tst = labels_tst.to(torch.float32).to(device)  
				v_tst = v_tst.to(torch.float32).to(device) 
				tokens_tst_idx = tokens_tst_idx.to(device)
				
				predicted_serie, _, labels_tst, tst_loss, _, _ = test_evaluation(model,test_batch, tokens_tst_idx, v_tst, labels_tst, windows_size, first_affect=False, sampling=False, number_of_samples=25, loss_kind='quantileloss_dist_normal')
				test_loss += tst_loss
				count+=1			
			test_loss = test_loss / count
			if num_epochs > 100 :
				if (epoch % 100 == 0 and epoch !=0):
					print(f'-- "test" loss {test_loss:.4}')
					print("------------------------")
			else :
				print(f'-- "test" loss {test_loss:.4}')
				print("------------------------")
		if test_loss < best_test_loss or epoch == 0:
			best_test_loss = test_loss
			torch.save({
			            'epoch': epoch,
			            'model_state_dict': model.state_dict(),
			            'optimizer_state_dict': opt.state_dict(),
			            'loss': loss,
			            }, PATH)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-directory', type=str, dest='data_directory', help='data directory', default='data_prepared/')
	parser.add_argument('--dataset-name', type=str, dest='dataset_name', help='Dataset name', default='data_prepared')
	parser.add_argument('--meta-information', type=str, dest='meta_information', help='meta information')
	parser.add_argument('--model-config', type=str, dest='model_config', help='model config')
	parser.add_argument('--train-config', type =str, dest = 'train_config', help ='train config')
	parser.add_argument('--model-save-path', type=str, dest='model_save_path', help='model save path')


	args = parser.parse_args()
	# Data LOading 
	# Directory and dataset name
	data_dir = args.data_directory
	dataset = args.dataset_name
	meta_information_path  = args.meta_information
	model_config_path  = args.model_config
	train_config  = args.train_config
	save_path = args.model_save_path

	main_train(data_dir,dataset,meta_information_path,model_config_path,train_config,save_path)