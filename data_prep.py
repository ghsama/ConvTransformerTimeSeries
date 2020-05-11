import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import argparse
import json

from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from pandas.tseries.frequencies import to_offset

from evaluate import *
import pickle

def gen_covariates(times):
	'''
	Function to create independent covariate variables and normalize them: weekday, hour, mouth

	# times : date times 

	returns : Normalized covariates variables
	'''
	num_covariates = 3
	covariates = np.zeros((times.shape[0], num_covariates))
	for i, input_time in enumerate(times):
		covariates[i, 0] = input_time.weekday()
		covariates[i, 1] = input_time.hour
		covariates[i, 2] = input_time.month
	return covariates[:, :num_covariates]

def prep_data(data, window_size, stride_size, data_start, timeseries_list, covariates=None, covariates_var=None, transformation_covariates_used = None, transformation_covariates_var_used = None):
	'''
	Function to prepare data for the model 
	
	# data : 2D array, contains the times series : e.g : (274,5) : 274 timestamps, 5 timeseries
	# window_size : int, the sliding windows size : e.g : 24
	# stride_size : int, the stride during the window_size sliding
	# covariates : 2D array, covariate variables, independents of the series : e.g (274, 3) : 274 timestamps, 3 covariate variables
	# covariates_var : 3D array, covariate variables that changes in time : e.g : (274,5,2) : 274 timestamps, 5 timeseries, 2 covariate variables
	# data_start : 1D array, the index in the timestamps where each timeseries start : e.g (5,)
	# timeseries_list : 1D array, the list of the names of the time series : e.G (5,)
	
	returns  : x_input, label, v_input, dict_timeseries
	
	# x_input :  3D array, contains the input which will be feeded to the model, combaining data and covariates : e.g (63,24,7) : 63 number of windows created from the sliding window on all the series, 24 the windows size, 7 : the variables : 1 (time serie value) + 3 (covariate value) + 2 (covariate_var value normalized)+ 1 (time serie id)
	# label : 2D array, contains the values of the time serie on the window shifted by one : e.g : (63, 24) : 63 : 63 number of windows created from the sliding window on all the series, 24 the window size
	# v_input : 2D array, the normalization factor by window : e.g : (63,2) : 63 , 2 : 1 (the normalizaiton factor) +1 (zeros)
	# dict_timeseries : dict, contains the series number which figure in x_input[:,:,-1] corresponding serie name
	'''
	# Initial global informations

	time_len = data.shape[0] # time series global lenght
	num_series = data.shape[1] # number of time series
	input_size = window_size-stride_size # Input size

	# Verification :

	if not covariates_var is None : 
		# Verification of index matching
		assert covariates_var.shape[0] == data.shape[0]
		# Verification of time series matching
		assert covariates_var.shape[1] == data.shape[1]

	if not covariates is None : 
		# Verification of index matching 
		assert covariates.shape[0] == data.shape[0]

	# keep only time series with at least one window size

	kept_timeseries = time_len - data_start > window_size 
	data = data[:,kept_timeseries] 
	data_start = data_start[kept_timeseries]
	covariates_var = None if covariates_var is None else covariates_var[:,kept_timeseries,:]
	timeseries_list = timeseries_list[kept_timeseries]
	
	
	# Update the parametres

	time_len = data.shape[0]-1  # minus the last one to use it as the la predsict
	num_series = data.shape[1]

	num_cov_simple = 0 if covariates is None else covariates.shape[1]
	num_cov_var = 0 if covariates_var is None else covariates_var.shape[2]

	num_covariates = num_cov_simple + num_cov_var

	dict_timeseries = dict(zip(timeseries_list,range(num_series)))

	# Number of Windows

	windows_per_series = np.full((num_series), (time_len-input_size) // stride_size) # number of windows in the interval
	windows_per_series -= (data_start+stride_size-1) // stride_size  

	# Data structre definition

	total_windows = np.sum(windows_per_series)
	
	x_input = np.zeros((total_windows, window_size, 1+num_covariates + 1), dtype='float32')
	label = np.zeros((total_windows, window_size), dtype='float32')
	v_input = np.zeros((total_windows, 2), dtype='float32')

	# transformation 

	if not covariates_var is None:
		transformation_covariates_var = dict()# np.zeros((num_series,num_cov_var,2)) StandardScaler
		covariates_var = covariates_var.astype(float)
	else :
		transformation_covariates_var = None
	
	if not covariates is None:
		transformation_covariates = dict() #np.zeros((num_series,num_cov_simple,2))
		covariates = covariates.astype(float)
	else :
		transformation_covariates = None

	## Scaling the covariates

	for i in range(num_cov_simple):
		if transformation_covariates_used is None and not covariates is None:
			# creating a scaler
			transfomer_cov_simple = StandardScaler()
			covariates[:,i] = transfomer_cov_simple.fit_transform(covariates[:,i].reshape(-1,1))[:,0]
			transformation_covariates[i] = transfomer_cov_simple
		elif not covariates is None :
			# loeading one
			transfomer_cov_simple = transformation_covariates_used[i]
			covariates[:,i] = transfomer_cov_simple.transform(covariates[:,i].reshape(-1,1))[:,0]
			transformation_covariates[i] = transfomer_cov_simple

	# Values assignation 
	
	count = 0
	for series in range(num_series):
		## Scaling the covariate variables
		if transformation_covariates_var_used is None and not covariates_var is None:
			# creating a scaler
			transfomer_cov_var = StandardScaler()
			covariates_var[data_start[series]:, series,:] = transfomer_cov_var.fit_transform(covariates_var[data_start[series]:, series,:])
		elif not covariates_var is None:
			# loading the used one for example in the training 
			transfomer_cov_var = transformation_covariates_var_used[series]
			covariates_var[data_start[series]:, series,:] = transfomer_cov_var.transform(covariates_var[data_start[series]:, series,:])
		
		if not covariates_var is None:
			transformation_covariates_var[series] = transfomer_cov_var

		# Windows creation 

		for i in range(windows_per_series[series]):
			# In training windowing begun at the start of the time serie

			window_start = stride_size*i+data_start[series]
			window_end = window_start+window_size

			# x_input shape is : timeserie_id, steps, timeserie_value+covariate_variables

			x_input[count, :, 0] = data[window_start:window_end, series] # timeserie_value, we start always by 0
			x_input[count, :, 1:1+num_cov_simple] = 0 if num_cov_simple==0 else covariates[window_start+1:window_end+1, :] # covariates_values
			x_input[count, :, 1+num_cov_simple:-1] = 0 if num_cov_var == 0 else covariates_var[window_start+1:window_end+1, series,:] # covariates than depend on the time serie
			x_input[count, :, -1] = series # timeserie id / label / token which the name is in "timeseries_list"

			# target affectation / label

			label[count, :] = data[window_start+1:window_end+1, series]

			# Time serie scale normalization using the window sum.
			
			nonzero_sum = (x_input[count, :, 0]!=0).sum()
			
			if nonzero_sum == 0:
				v_input[count, 0] = 1
			else:
				v_input[count, 0] = np.true_divide(x_input[count, :, 0].sum(),nonzero_sum)+1
				x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
				label[count, :] = label[count, :]/v_input[count, 0]
			count += 1
	return x_input, label, v_input, dict_timeseries, transformation_covariates_var, transformation_covariates

def prep_data_for_prediction(model, data, window_size, steps, data_start, timeseries_list,device = 'cpu' , covariates=None, covariates_var=None, transformation_covariates_used = None, transformation_covariates_var_used = None, sampling = False):
	'''
	Function to prepare data for the model 
	# model : the pytorch Transformer model to use in prediction
	# data : 2D array, contains the times series : e.g : (274,5) : 274 timestamps, 5 timeseries
	# window_size : int, the sliding windows size : e.g : 24
	# stride_size : int, the stride during the window_size sliding
	# steps : int , steps to predict which needs to be < window_size
	# covariates : 2D array, covariate variables, independents of the series : e.g (274, 3) : 274 timestamps, 3 covariate variables
	# covariates_var : 3D array, covariate variables that changes in time : e.g : (274,5,2) : 274 timestamps, 5 timeseries, 2 covariate variables
	# data_start : 1D array, the index in the timestamps where each timeseries start : e.g (5,)
	# timeseries_list : 1D array, the list of the names of the time series : e.G (5,)
	# transformation_covariates_used : used Transformation during train
	# transformation_covariates_var_used : used Transformation during train
	# sampling : Boolean, for using the mean of the distribution or doing a sampling
	
	returns  : estimation_modeles
	
	# estimation_model : 2D array, containing the predictions for the series
	'''
	
	# Stride is 1 and not variable in the prediction mode

	stride_size = 1
	
	# torch device
	if device == None :
		if torch.cuda.is_available():
			torch.set_default_tensor_type(torch.cuda.FloatTensor)
			device = 'cuda'
	# Initial global informations

	time_len = data.shape[0] # time series global lenght
	num_series = data.shape[1] # number of time series
	input_size = window_size-stride_size # Input size

	# Verification for prediction:

	assert steps < window_size, 'The steps='+str(steps)+' is > window_size='+str(window_size)
	assert steps > 0, 'The steps='+str(steps)+' needs to be > 0'

	if not covariates_var is None : 
		assert not transformation_covariates_var_used is None, 'YOU NEED TRANSFMATION FOR THE COVARIATES VAR'
		# Verification of index matching
		assert covariates_var.shape[0] == data.shape[0] + steps, 'covariates_var.shape[0] '+str(covariates_var.shape[0])+' and data.shape[0] + steps '+str(data.shape[0] + steps)+' didn t match'
		# Verification of time series matching
		assert covariates_var.shape[1] == data.shape[1]

	if not covariates is None : 
		assert not transformation_covariates_used is None, 'YOU NEED TRANSFMATION FOR THE COVARIATES'
		# Verification of index matching 
		assert covariates.shape[0] == data.shape[0] + steps, 'covariates.shape[0] '+str(covariates.shape[0])+' and data.shape[0] + steps '+str(data.shape[0] + steps)+' didn t match'
	
	# Append -1 to the new steps to predict to the predict
	# keep only time series with at least one window size

	kept_timeseries = time_len - data_start > window_size 
	data = data[:,kept_timeseries] 
	data_start = data_start[kept_timeseries]
	covariates_var = None if covariates_var is None else covariates_var[:,kept_timeseries,:]
	timeseries_list = timeseries_list[kept_timeseries]

	# Number of series to predict : 
	q = 100
	data = data[:,:q]
	data_start = data_start[:q]
	covariates_var = None if covariates_var is None else covariates_var[:,:q,:]

	# Update the parametres

	time_len = data.shape[0]
	num_series = data.shape[1]

	num_cov_simple = 0 if covariates is None else covariates.shape[1]
	num_cov_var = 0 if covariates_var is None else covariates_var.shape[2]
	
	num_covariates = num_cov_simple + num_cov_var

	dict_timeseries = dict(zip(timeseries_list,range(num_series)))

	# Number of Windows

	windows_per_series = np.full((num_series), (time_len-input_size) // stride_size) # number of windows in the interval
	windows_per_series -= (data_start+stride_size-1) // stride_size  

	# Data structre definition

	total_windows = np.sum(windows_per_series)
	

	# transformation 
	if not covariates_var is None : 
		transformation_covariates_var = dict()# np.zeros((num_series,num_cov_var,2)) StandardScaler
	else : 
		transformation_covariates_var = None

	if not covariates is None : 
		transformation_covariates = dict() #np.zeros((num_series,num_cov_simple,2))
	else : 
		transformation_covariates = None

	## Scaling the covariates
	
	for i in range(num_cov_simple):
		if transformation_covariates_used is None:
			# creating a scaler
			transfomer_cov_simple = StandardScaler()
			covariates[:,i] = transfomer_cov_simple.fit_transform(covariates[:,i].reshape((-1,1)))[:,0]
			transformation_covariates[i] = transfomer_cov_simple
		else :
			# creating a scaler
			transfomer_cov_simple = transformation_covariates_used[i]
			covariates[:,i] = transfomer_cov_simple.fit_transform(covariates[:,i].reshape((-1,1)))[:,0]
			transformation_covariates[i] = transfomer_cov_simple

	# Values assignation
	data = np.append(data, -1*np.ones((steps,num_series)), axis=0) # add the steps to predict
	estimation_model = np.empty((data.shape[0]-1,data.shape[1]))
	sigma_model = np.empty((data.shape[0]-1,data.shape[1]))
	
	count = 0
	print('# Device :', device)
	for series in range(num_series):
		## Scaling the covariate variables
		if transformation_covariates_var_used is None and not covariates_var is None:
			# creating a scaler
			transfomer_cov_var = StandardScaler()
			covariates_var[data_start[series]:, series,:] = transfomer_cov_var.fit_transform(covariates_var[data_start[series]:, series,:])
			transformation_covariates_var[series] = transfomer_cov_var
		elif not covariates_var is None:
			# creating a scaler
			transfomer_cov_var = transformation_covariates_var_used[series]
			covariates_var[data_start[series]:, series,:] = transfomer_cov_var.transform(covariates_var[data_start[series]:, series,:])
			transformation_covariates_var[series] = transfomer_cov_var

		# Windows creation 
		window_start = data_start[series]
		i = 0
		
		while stride_size*i + data_start[series] + window_size < time_len + steps - 1 - data_start[series] :  # time_len + steps - 1 to match the new data size - 1
			# In training windowing begun at the start of the time serie

			window_start = stride_size*i + data_start[series]
			window_end = window_start + window_size

			x_input = np.zeros((1, window_size, 1 + num_covariates + 1), dtype='float32')
			
			# x_input shape is : timeserie_id, steps, timeserie_value+covariate_variables

			x_input[0, :, 0] = data[window_start:window_end, series] # timeserie_value, we start always by 0
			x_input[0, :, 1:1+num_cov_simple] = 0 if num_cov_simple==0 else covariates[window_start + 1:window_end + 1, :] # covariates_values
			x_input[0, :, 1+num_cov_simple:-1] = 0 if num_cov_var == 0 else covariates_var[window_start + 1:window_end + 1, series,:] # covariates than depend on the time serie
			x_input[0, :, -1] = series # timeserie id / label / token which the name is in "timeseries_list"

			# Time serie scale normalization using the window sum

			v_input = np.zeros((1, 2), dtype='float32')

			nonzero_sum = (x_input[0, :, 0]!=0).sum()
			
			if nonzero_sum == 0:
				v_input[0, 0] = 1
			else:
				v_input[0, 0] = np.true_divide(x_input[0, :, 0].sum(),nonzero_sum)+1
				x_input[0, :, 0] = x_input[0, :, 0]/v_input[0, 0]

			# To tensor 
			v_input = torch.tensor(v_input)
			x_input = torch.tensor(x_input)

			# Expanding v_input
			v_input_expanded = v_input.expand(window_size,1,2).transpose(1,0)

			# tocken
			token = np.array([series])
			token = torch.tensor(token)

			# model application
			x_input = x_input[:,:,:-1] # since the token info is given separately 
			
			x_input = x_input.to(device)
			v_input_expanded = v_input_expanded.to(device)
			token = token.to(device)
			
			if sampling :
				samples, estimation, mu, sigma = evaluate(model, x_input, token, v_input_expanded, window_size, sampling, number_of_samples = 100, scaled_param = True)
			else :
				estimation, mu, sigma = evaluate(model, x_input, token, v_input_expanded, window_size, sampling, number_of_samples = 100, scaled_param = True) 			

			estimation_model[window_start + 1:window_end + 1, series] = estimation[0,:,0].detach().cpu().numpy()
			sigma_model[window_start + 1:window_end + 1, series] = sigma[0,:,0].detach().cpu().numpy()

			if not window_end < time_len :
				data[window_end,series] = estimation[0,-1,0].detach().cpu().numpy()
			i+=1
			count += 1

	return estimation_model, sigma_model

def save_dataset(save_path,save_name,xtrain_input,v_input_train,label_train,timeseries_dict_train,transformation_covariates_var, transformation_covariates,xtest_input,v_input_test,label_test,timeseries_dict_test, split_dates):
	'''
	Function to save the prepared data
	# save_path
	# save_name
	#....

	'''
	# saving train
	prefix = os.path.join(save_path, 'train_')

	np.save(prefix+'data_'+save_name, xtrain_input)
	np.save(prefix+'v_'+save_name, v_input_train)
	np.save(prefix+'label_'+save_name, label_train)
	np.save(prefix+'timeseries_dict_'+save_name, timeseries_dict_train)

	# saving test
	prefix = os.path.join(save_path, 'test_')
	np.save(prefix+'data_'+save_name, xtest_input)
	np.save(prefix+'v_'+save_name, v_input_test)
	np.save(prefix+'label_'+save_name, label_test)
	np.save(prefix+'timeseries_dict_'+save_name, timeseries_dict_test)

	# savinf transformation
	with open(save_path+'/transformation_covariates_var_'+save_name+'.pkl', 'wb') as file:
		pickle.dump(transformation_covariates_var, file, pickle.HIGHEST_PROTOCOL)

	with open(save_path+'/transformation_covariates_'+save_name+'.pkl', 'wb') as file:
		pickle.dump(transformation_covariates, file, pickle.HIGHEST_PROTOCOL)

	# saving meta informations
	num_timeseries_kinds = len(timeseries_dict_train)
	
	window_size = xtrain_input.shape[1]
	number_of_vars = xtrain_input.shape[2]-1 # -1 cz the last variable is the kind of the time serie

	meta_information = dict({
		'num_timeseries_kinds' : num_timeseries_kinds,
		'window_size' : window_size,
		'number_of_vars' : number_of_vars,
		})
	train_start,train_end,test_start,test_end = split_dates

	meta_information['transformation_covariates_var_path'] = save_path+'/transformation_covariates_var_'+save_name+'.pkl'
	meta_information['transformation_covariates'] = save_path+'/transformation_covariates_'+save_name+'.pkl'

	meta_information['train_start'] = str(train_start)
	meta_information['train_end'] = str(train_end)
	meta_information['test_start'] = str(test_start)
	meta_information['test_end'] = str(test_end)

	with open(save_path+'/meta_information.json', 'w') as file:
		json.dump(meta_information, file)

	# Creation of the default model configuration && default train config
	model_config = dict()
	model_config['headers'] = 3
	model_config['depth'] = 3
	model_config['kernel_size'] = 6
	model_config['default'] = True

	config_folder_path = os.path.join('models',save_name)
	os.makedirs(config_folder_path, exist_ok=True)

	config_path = os.path.join(config_folder_path,'model_config.json')
	
	with open(config_path, 'w') as file:
		json.dump(model_config, file)

	# Creation of the default model configuration
	train_config = dict()
	train_config['num_epochs'] = 5
	train_config['lr_warmup'] = 1000
	train_config['learning_rate'] = 0.001
	train_config['default'] = True
	train_config['batch_size']  = 32
	train_config['predict_batch_size'] = 32

	train_path = os.path.join(config_folder_path,'train_config.json')
	with open(train_path, 'w') as file:
		json.dump(train_config, file)

def generate_fake_data(interval= ['2012-12-31','2015-07-29']):
	'''
	Function to generate a fake test data
	'''
	# Main time serie to predict
	Table = pd.DataFrame(pd.date_range(interval[0],interval[1]), columns=['date'])

	num_tot = len(Table)
	serie_normal = []
	serie_sin_normal = []
	for i in range(num_tot):
		serie_normal.append(np.random.normal(2*i,scale=50)) #np.sin(200*np.pi*i/num_tot)*10+
		serie_sin_normal.append(np.random.normal(2*i,scale=50)+np.sin(200*np.pi*i/num_tot)*500)
							
	plt.plot(serie_normal)
	plt.plot(serie_sin_normal)
	plt.show()

	Table['serie_normal'] = serie_normal
	Table['serie_sin_normal'] = serie_sin_normal
	Table.set_index('date',inplace=True)

	return Table

def generate_fake_data_covariates(interval= ['2012-12-31','2015-07-29']):
	'''
	Function to generate a fake test data
	'''
	covariates_var = []
	# var 1
	# Main time serie to predict
	Table = pd.DataFrame(pd.date_range(interval[0],interval[1]), columns=['date'])

	num_tot = len(Table)
	serie_cos_normal = []
	for i in range(num_tot):
		serie_cos_normal.append(np.random.normal(2*i,scale=50)+np.cos(200*np.pi*i/num_tot)*500) #np.sin(200*np.pi*i/num_tot)*10+
							
	plt.plot(serie_cos_normal)
	plt.show()

	Table['serie_cos_normal'] = serie_cos_normal
	Table.set_index('date',inplace=True)

	# Adding to the covariates_var
	covariates_var.append(Table)

	# var 2
	Table = pd.DataFrame(pd.date_range(interval[0],interval[1]), columns=['date'])

	num_tot = len(Table)
	serie_tan_normal = []
	for i in range(num_tot):
		serie_tan_normal.append(np.random.normal(2*i,scale=50)+np.tan(200*np.pi*i/num_tot)*500) #np.sin(200*np.pi*i/num_tot)*10+
							
	plt.plot(serie_tan_normal)
	plt.show()

	Table['serie_tan_normal'] = serie_tan_normal
	Table.set_index('date',inplace=True)

	# Adding to the covariates_var
	covariates_var.append(Table)

	return covariates_var

def main(data_path,data_covariates_path,date_column,cast_float16,window_size,stride_size,test_ratio,save_name):

	# Data loading
	if data_path :
		print('## Data reading :')
		if date_column :
			Table = pd.read_csv(data_path)
			Table[date_column] = pd.to_datetime(Table[date_column])
			Table.set_index(date_column,inplace=True)
		else : 
			Table = pd.read_csv(data_path, index_col=0, parse_dates=True)

			if cast_float16 :
				Table = Table.astype('float16')
			inf_index = Table.index[np.isinf(Table).any(1)]
			inf_col =  Table.columns.to_series()[np.isinf(Table).any()]
	else :
		print('## Data generation :')
		Table = generate_fake_data()

	if data_covariates_path:
		print('## Data covariates reading :')
		covariates_var = []
		for file in os.listdir(data_covariates_path):
			if file.endswith(".csv"):
				if date_column :
					var = pd.read_csv(data_covariates_path+'/'+file)
					var[date_column] = pd.to_datetime(var[date_column])
					var.set_index(date_column, inplace=True)
				else :
					var = pd.read_csv(data_covariates_path+'/'+file, index_col = 0, parse_date=True)
				covariates_var.append(var)
	else :
		covariates_var = None
		if not data_path :
			covariates_var = generate_fake_data_covariates()

	print('# Preparing the data :')
	
	# Global param

	print("\t # Window_size :",window_size)
	print("\t # Stride_size :",stride_size)
	print("\t # test_ratio :",test_ratio)

	l_total = len(Table)

	# Train and Test split
	train_start = Table.index[0]
	train_end = Table.index[int(l_total*(1-test_ratio))]

	freq = pd.infer_freq(Table.index)
	
	freq = to_offset(freq)
	
	test_start = train_end - freq*8 #Table.index[int(l_total*(1-test_ratio))]
	test_end = Table.index[-1]

	# List of time series
	timeseries_list = np.array(list(Table.columns))

	train_data = Table[train_start:train_end].values
	test_data = Table[test_start:test_end].values

	print('\t #train_data index range:',train_start,train_end)
	print('\t #test_data index range:',test_start,test_end)

	# Covariate variables preparation

	print('# Covariate variables preparation :')
	if not covariates_var is None : 
		train_covariates_var = []
		for var in covariates_var :
			train_covariates_var.append(var[train_start:train_end].values[:,:,None])
		train_covariates_var = np.concatenate(train_covariates_var,axis=2)

		test_covariates_var = []
		for var in covariates_var :
			test_covariates_var.append(var[test_start:test_end].values[:,:,None])
		test_covariates_var = np.concatenate(test_covariates_var,axis=2)
	else :
		train_covariates_var = None
		test_covariates_var = None

	## Train covariates
	covariates_train = gen_covariates(Table[train_start:train_end].index)
	## Test covariates
	covariates_test = gen_covariates(Table[test_start:test_end].index)

	# data_start
	data_start_train = (train_data!=0).argmax(axis=0) #find first nonzero value in each time series
	data_start_test = (test_data!=0).argmax(axis=0) #find first nonzero value in each time series

	# Apply the transformation
	print('# Transforming the data :')

	window_size_train = window_size
	window_size_test = window_size

	xtrain_input, label_train, v_input_train, timeseries_dict_train, transformation_covariates_var, transformation_covariates = prep_data(train_data, window_size=window_size_train, stride_size=stride_size, data_start=data_start_train, timeseries_list=timeseries_list, covariates=covariates_train, covariates_var=train_covariates_var)
	xtest_input, label_test, v_input_test, timeseries_dict_test, _, _ = prep_data(test_data, window_size=window_size_test, stride_size=stride_size, data_start=data_start_test, timeseries_list=timeseries_list, covariates=covariates_test, covariates_var=test_covariates_var,transformation_covariates_used = transformation_covariates, transformation_covariates_var_used = transformation_covariates_var)

	# Saving 
	save_path = 'data_prepared/'
	os.makedirs(save_path+save_name, exist_ok = True)
	save_path = save_path+save_name
	
	print('# Saving th data .')
	print('\t # Used save_name:',save_name)
	save_dataset(save_path,save_name,xtrain_input,v_input_train,label_train,timeseries_dict_train,transformation_covariates_var, transformation_covariates,xtest_input,v_input_test,label_test,timeseries_dict_test, [train_start,train_end,test_start,test_end])

if __name__ == "__main__":
	print('# Loading the data :')

	# Parsing the arguments 
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-path', type=str, dest='data_path', help='data csv path', default=None)
	parser.add_argument('--data-covariates-path', type=str, dest='data_covariates_path', help='data csv path', default=None)
	parser.add_argument('--time-column-name', type=str, dest='date_column', help='timme column name', default=None)
	parser.add_argument('--save-name', type=str, dest='save_name', help='Save name', default='data_prepared')

	parser.add_argument('--window-size', type=int, dest='window_size', help='window_size', default=100)
	parser.add_argument('--stride-size', type=int, dest='stride_size', help='stride_size', default=4)
	parser.add_argument('--test-ratio', type=float, dest='test_ratio', help='test_ratio', default=0.30)

	parser.add_argument('--cast-float16', type=bool, dest='cast_float16', help='cast_float16', default=False)

	args = parser.parse_args()

	# Extracting arguments :
	data_path = args.data_path
	data_covariates_path = args.data_covariates_path
	date_column = args.date_column
	cast_float16 = args.cast_float16

	window_size = args.window_size
	stride_size = args.stride_size
	test_ratio = args.test_ratio
	save_name = args.save_name

	# Main
	main(data_path,data_covariates_path,date_column,cast_float16,window_size,stride_size,test_ratio,save_name)

