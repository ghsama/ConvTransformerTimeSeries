# Convolution Transformer :
This is first implimentation of 1D convolutional transformer for time series, it is inspired from the article [Enhancing the Locality and Breaking the MemoryBottleneck of Transformer on Time Series Forecasting](https://arxiv.org/pdf/1907.00235.pdf) .

The model consists of a stack of transformers which takes as input the a window of instance and predict the distribution of the next value.

The 1D convolution aims to have a better similarity measure in the attention layer.

The used loss is a Quantile loss or a KL loss (depends on the choice you made).

# The framwork : 

Giving a list of time series (e.g : electricity consumation in many areas) and a corresponding covariate variables (e.g : weather temperature, precepitation, holidays ..) we aim to predict the next instance distribution (the mean and the standard deviation of a Log Normal).

# Usage :
Applying the models on the electricity dataset used in the article.
Downloading the dataset : 
    python elect_preprocessing.py

Preprocessing the data : 

    python data_prep.py --data-path data_raw/elect/LD2011_2014.txt --save-name elect --window-size 100 --stride-size 500 --test-ratio 0.3

Launching the training :

    python train.py --data-directory data_prepared/elect/ --dataset-name elect --meta-information data_prepared/elect/meta_information_elect.json --model-config models/elect/model_config.json --train-config models/elect/train_config.json --model-save-path models/

Extracting a portion of the test data to infeer :
    
    python elect_test.py

Applying the prediction :

    python prediction.py --data-path data_raw/elect_prediction/500_LD2011_2014.txt --save-path predictions --steps 1 --model-config models/elect/model_config.json --meta-information data_prepared/elect/meta_information.json --prediction-name elect_01

Resulats :

The vertical line is the limit between train and test .

![alt text](predictions/elect_01/11.png?raw=true)
![alt text](predictions/elect_01/20.png?raw=true)
![alt text](predictions/elect_01/15.png?raw=true)
![alt text](predictions/elect_01/44.png?raw=true)

# Steps Explanation :

**Preprocessing:** 
*data_prep.py* : 
Preprocessing time series data and covariates and transforming them in the appropriate form.

    - data.csv file : a file containing in columns all the time series kinds and in the lines the time steps + a time column (The instance before the start oof the time series may be filled with 0 )
    - covariate_folder : a folder containing as many CSVs as the covariate variables we have, with eath one having the same structure as the data.csv
    
    + time series independent covariates : generated automaticlly (e.g : the mounth/ the day corresponding to the timestep)
Arguments : 
    
    --data-path : path to the csv file : it may contain many columns, each for a kind of time series + a time column
    --data-covariates-path : path to the directory containing the  covariate variables dataframes, it needs to have the same number of time series (columns) as in the data-path csv.
    --time-column-name : the column which contains the timesteps
    --save-name : the name used for the preprocessed data saving

    --window-size : the window size used in the training and the prediction
    --stride-size : stride between the generated windows for the training 
    --test-ratio : the ratio of the test data

Outputs : In */data_prepared/save_name/* directory

    - train and test npy arrays preprocessed
    - meta_information json : number of time series, window_size and number of covariates
    - covariate normalisation transformations
    
**Training:**
*train.py* : 
Creating and training a Transformers blocks model with a Quantile/KL loss on a LogNormal/Binomial

    - directory of the preprocessed train and test data 
    - model configuration and training configuration

Arguments :

    --data-directory : the path to the directory of the saved preprocesed data (*/data_prepared/save_name/* )
    --dataset-name : the save name used during the preprocessing (*save_name*)
    --meta-information : the path to the jason of the meta_information (*/data_prepared/save_name/meta_information_save_name.json* )
    --model-config : path to json which contains the model architecture :
        Number of heads in the transformer
        Number of the transformers blocks 
        1d convolution kernel size
        e.g : {"headers": 3, "depth": 3, "kernel_size": 6}
    --train-config : path to json which contains the training parameters :
        batch_size : the batch size during training
        predict_batch_size : the batch size during prediction test
        num_epochs : the number of epochs
        lr_warmup : the number of warmup
        learning_rate : the learning rate
        e.g : {"batch_size": 32, "predict_batch_size": 32, "num_epochs": 7, "lr_warmup":1000, "learning_rate":0.001}
    --model-save-path : path to directory where to save the model and the configuration json.

Outputs : In *save_path/* directory

    - the configuration of the created model json file:
        k : Number of time series + number of covriates (the static ones -e.g : day/hour- and dynamic ones -e.g : temperature)
    	headers : Number of heads in the transformers
    	depth : Number of the transformers blocks 
    	seq_length : window size
    	kernel_size : 1d convolution kernel size used in the attention layers inside the transformers blocks
    	num_tokens : Number of time series kinds used in the train (the number of columns in the data.csv )
    	path : path to the model inside *save_path/*
    - best trained model : checkpoint of the model at the lowest test loss
**Inference:**
*prediction.py* :
Making inference for future  steps 
    
    - trained model with its own configuration
    - data transformers
    - time series to infeer + their coressponding covariate variables

Arguments :

    --data-path : path to the prediction table with same columns as in the training
    --data-covariates-path : path to the directory that contains the covariates tables, having *steps* more timestamps than the data Table
    --time-column-name : the column which contains the timesteps
    --save-path : directory path to save in
    --steps : number od steps to predict (!!! NORMALY need to be 1, but you can inferre for more with the hypothesis that the prediction you make are 100% accurate)
    --model-config : path to model config path, used in the training
    --meta-information : path to the meta data used in the training 
    --prediction-name : folder name used to save the predictions
    --flag-real-plot : plot the real values (need to have steps == 1)

    
Outputs :  In *save_path/prediction_name/* directory

    - estimation : which is the model estimations
    - sigma : the standard deviation of the predictions
    - plots of the predictions

# Sepecial thanks to : 
  - Yunkai Zhang for the implimentation of [DeepAR](https://github.com/zhykoties/TimeSeries), which i used to get some inspiration.
