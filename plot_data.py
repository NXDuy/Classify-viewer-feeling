import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
    
    location_list = using_data.groupby(['venue', "viewer feeling of youtuber's style "], as_index=True).size()
    return location_list

FILE_DIR = './Data/Data_AIL.xlsx'
COLUMNS = ['start time', 'end time', 'venue',
            "viewer feeling of youtuber's style "]
loc_detail = read_file(FILE_DIR, COLUMNS)
index = list(loc_detail.index)

print(loc_detail)

# ax1.bar3d(x_axis, y_axis, bottom, width, depth, z_axis, shade=True)
# ax1.set_title('Shaded')

# plt.show()