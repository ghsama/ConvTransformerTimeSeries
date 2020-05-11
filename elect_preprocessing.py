from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import os
import pandas as pd


if __name__ == '__main__':
save_name = 'elect'
name = 'LD2011_2014.txt'
save_path = os.path.join('data_raw', save_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

csv_path = os.path.join(save_path, name)
if not os.path.exists(csv_path):
    zipurl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'
    print('Start download')
    with urlopen(zipurl) as zipresp:
        print('Start unzip')
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(save_path)
print('Treating and saving')
data_frame = pd.read_csv(csv_path, sep=";", index_col=0, parse_dates=True, decimal=',')
#data_frame = data_frame.resample('1H',label = 'left',closed = 'right').sum()
data_frame.fillna(0, inplace=True)
data_frame.to_csv(csv_path)