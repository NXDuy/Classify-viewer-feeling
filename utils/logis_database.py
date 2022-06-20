
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)


from utils.database import read_file
from torch.utils.data import Dataset, DataLoader
import torch

FILE_DIR = './Data/Data_AIL.xlsx'
COLUMNS = ['start time', 'end time', 'Unnamed: 11',
        "viewer feeling of youtuber's style ", 'describe how to make it', 'venue']

class FeelingClassify(Dataset):
    def __init__(self, file_dir=FILE_DIR, columns=COLUMNS, has_bias=True):
        super(FeelingClassify, self).__init__()
        self.input_data, self.output_data = read_file(file_dir, columns)
        # print(self.output_data)
        self.n_samples = self.output_data.shape[0]
        self.n_features = self.input_data.shape[1]
           
        self.__modify(ouput_data=self.output_data, num_classes=6)
        if has_bias == True:
            # print(self.input_data.shape)
            self.input_data = torch.cat((self.input_data, torch.ones(self.n_samples, 1)),1)
            # print(self.input_data)
            self.n_features += 1
        
    def __modify(self, ouput_data, num_classes):
        
        modify_output = torch.zeros(ouput_data.shape[0], num_classes)
        for n_sample, sample in enumerate(ouput_data):
            # print(sample.item())
            modify_output[n_sample, int(sample.item())].add_(1)
        
        self.output_data = modify_output



    def __getitem__(self, index):
        return self.input_data[index], self.output_data[index]

    def __len__(self):
        return self.n_samples

# dataset = FeelingClassify()
# data_loader = DataLoader(dataset=dataset, shuffle=True)

# for input, output in data_loader:
    # print(input, output)
    # break