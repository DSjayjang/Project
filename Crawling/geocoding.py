import os
import time
import numpy as np
import pandas as pd

from utils import geocoding

# data load
df = pd.read_excel(r'.\datasets\250716.xlsx')

num = np.array(df['순번'])
start = np.array(df['출발지 주소'])
mid = np.array(df['중간집결지 주소'])
end = np.array(df['최종 도착지 주소'])

# geocoding
slc = slice(0, 3)

print('Processing...')
s = time.time()

start_lat, start_lng = zip(*(geocoding(addr) for addr in start[slc]))
mid_lat, mid_lng = zip(*(geocoding(addr) for addr in mid[slc]))
end_lat, end_lng = zip(*(geocoding(addr) for addr in end[slc]))

e = time.time()
print('Done')

print(f'걸린시간(초): {e-s:.2f}')

df_new = pd.DataFrame({'num': num[slc],
                       'start_lat': start_lat,
                       'start_lng': start_lng,
                       'mid_lat': mid_lat,
                       'mid_lng': mid_lng,
                       'end_lat': end_lat,
                       'end_lng': end_lng})
df_new