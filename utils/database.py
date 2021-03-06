import os
import sys
from xml.dom.pulldom import default_bufsize
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

# from pickletools import uint1
import torch
import pandas as pd
import re
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder

def read_file(file_dir, columns=None):
    try:
        raw_data = pd.read_excel(file_dir)
    except:
        print('CANNOT READ FILE DATA')
        quit(0)
        # return None
    
    # print(raw_data.columns)
    using_data = raw_data[columns]
    using_data = using_data.dropna(axis=0)
    
    index_data = list()
    for str_time in using_data['start time']:
        if str_time[0].isdigit() and str_time[-1].isdigit():
            index_data.append(True)
        else:
            index_data.append(False)
    
    using_data = using_data[index_data]

    index_data = list()
    for str_time in using_data['end time']:
        if str_time[0].isdigit() and str_time[-1].isdigit():
            index_data.append(True)
        else:
            index_data.append(False)

    using_data = using_data[index_data]
    df_time = list()
    for start_time, end_time in zip(using_data['start time'], using_data['end time']):
        start_time_in_seconds = process_time_str(start_time)
        end_time_in_seconds = process_time_str(end_time)
        if start_time_in_seconds > end_time_in_seconds:
            start_time_in_seconds, end_time_in_seconds = end_time_in_seconds, start_time_in_seconds
        df_time.append((start_time_in_seconds, end_time_in_seconds))
    
    using_data[['start time', 'end time']] = df_time
    using_data["viewer feeling of youtuber's style "] = using_data["viewer feeling of youtuber's style "].astype('int') 
    using_data['describe how to make it'] = using_data['describe how to make it'].astype('int')

    # Normalization
    using_data['start time'] = using_data['start time']/using_data['start time'].max()
    using_data['end time'] = using_data['end time']/using_data['end time'].max()
    using_data['Unnamed: 11'] = using_data['Unnamed: 11']/using_data['Unnamed: 11'].max()
    
    # Label encoding
    label_encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_data = pd.DataFrame(label_encoder.fit_transform(using_data[['venue']]).toarray())
    using_data = using_data.join(encoder_data)
    using_data.drop(columns=['venue'], inplace=True)

    # Rename columns
    default_columns = ['start time', 'end time', 'Unnamed: 11', "viewer feeling of youtuber's style ", 'describe how to make it']
    columns = default_columns + list(label_encoder.categories_[0])
    using_data.columns = columns
    using_data = using_data.dropna(axis=0)

    input_data = using_data.loc[:, using_data.columns != "viewer feeling of youtuber's style "]
    output_data = using_data[["viewer feeling of youtuber's style "]]
    
    # print(output_data)

    # using_data.to_csv('Feeling.csv')
    # print(input_data.tail())
    # print(output_data.columns)
    # print(using_data.dtypes)
    return torch.tensor(input_data.values, dtype=torch.float32), torch.tensor(output_data.values, dtype=torch.float32)
        
def __check_value_time(hours=0, minutes=0, seconds=0):
    # print('Time out: 'hours, minutes, seconds)
    if hours > 24:
        print('Time error: ',hours, minutes, seconds)
        raise ValueError('Hour is out of range')

    if seconds > 60:
        print('Time error: ',hours, minutes, seconds)
        raise ValueError('Second is out of range')

    if minutes > 60:
        print('Time error: ',hours, minutes, seconds)
        raise ValueError('Minutes is out of range')

def process_time_str(time_str):
    time_data = re.findall('[0-9]?[0-9]', time_str)
    TIME_WITH_HOUR = 3
    TIME_WITH_MINUTES = 2
    TIME_WITH_SECONDS = 1
    MAX_SECONDS = 90060
    # print(time_data)
    if len(time_data) == TIME_WITH_HOUR:
        hours = int(time_data[0])
        minutes = int(time_data[1])
        seconds = int(time_data[2])

        __check_value_time(hours, seconds, minutes)

        return (hours*3600 + minutes*60 + seconds)
    if len(time_data) == TIME_WITH_MINUTES:
        minutes = int(time_data[0])
        seconds = int(time_data[1])

        __check_value_time(minutes=minutes, seconds=seconds)

        return (minutes*60 + seconds)
    if len(time_data) == TIME_WITH_SECONDS:
        seconds = int(time_data[0])

        __check_value_time(seconds=seconds)

        return seconds

def lr_schedular(cur_epoch, lr, lr_decay, epoch_decay):
    if cur_epoch%epoch_decay == 0:
        return lr*(1-lr_decay)
    else: 
        return lr

FILE_DIR = './Data/Data_AIL.xlsx'
COLUMNS = ['start time', 'end time', 'Unnamed: 11',
            "viewer feeling of youtuber's style ", 'describe how to make it', 'venue']

class FoodFeeling(Dataset):
    def __init__(self, file_dir=FILE_DIR, columns=COLUMNS, has_bias=True):
        super(FoodFeeling, self).__init__()
        self.input_data, self.output_data = read_file(file_dir, columns)
        self.n_samples = self.output_data.shape[0]
        self.n_features = len(columns) - 1
        if has_bias == True:
            # print(self.input_data.shape)
            self.input_data = torch.cat((self.input_data, torch.ones(self.n_samples, 1)),1)
            # print(self.input_data)
            self.n_features += 1

    def __getitem__(self, index):
        return self.input_data[index], self.output_data[index]

    def __len__(self):
        return self.n_samples

# input, output = read_file(FILE_DIR, COLUMNS)
# print(output)