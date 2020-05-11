import pandas as pd
import os
import json

data_path = 'data_raw/elect/LD2011_2014.txt'
data = pd.read_csv(data_path, index_col=0, parse_dates=True)


meta_information_path = 'data_prepared/elect/meta_information.json'
with open(meta_information_path, 'r') as file: 
	meta_information = json.load(file)

test_start = meta_information['test_start']

loc_end = data.index.get_loc(test_start)
data_test = data.iloc[loc_end-100:]

os.makedirs('data_raw/elect_prediction/',exist_ok=True)
data_test.iloc[:400].to_csv('data_raw/elect_prediction/500_LD2011_2014.txt')
