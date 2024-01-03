import numpy as np
import pandas as pd
import os
from PIL import Image

# Opening the data dor R2-R3
data_box_12 = pd.read_excel('/Users/filruff/Documents/GitHub/COVID19-ItaChina/data/AIforCOVID/processed/box_data_AXF12.xlsx')

data_box_1 = data_box_12[[False if '_1_' in id_ or '_2_' in id_ else True for id_ in data_box_12['img'].to_list()]].drop(columns=['Unnamed: 0'])
data_box_1.to_excel('/Users/filruff/Documents/GitHub/COVID19-ItaChina/data/AIforCOVID/processed/box_data_AXF1.xlsx')



print('here')


