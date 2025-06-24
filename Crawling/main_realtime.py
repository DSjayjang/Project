import os
import time
import random
import pandas as pd

from config import CHROM_DRIVER_PATH, WAIT_TIMEOUT, DELAY_RANGE
from utils import CoordinateTransformer
from crawler import RealtimeCrawling

# Data Load
os.chdir(r'C:\Users\user\Desktop\연구\5. 국방부 용역과제') # pcrl
# os.chdir(r'C:\Users\linde\OneDrive\Desktop\3. 연구\5. 국방부 용역과제') # my pc
# os.chdir(r'C:\Users\linde\Desktop\연구\5. 국방부 용역') # laptop
df = pd.read_csv('DB.csv')

# 출발지/목적지 위/경도
"""
from_lat, from_lon: list
: 출발지 위도, 경도
to_lat, to_lon: list
: 목적지 위도, 경도
"""
from_lat = df['Latitude'][:3].to_numpy()
from_lon = df['Longitude'][:3].to_numpy()
to_lat = df['Latitude'][4:7].to_numpy()
to_lon = df['Longitude'][4:7].to_numpy()

# 1. 좌표 변환
ct = CoordinateTransformer()
from_x, from_y = ct.transform(from_lon, from_lat)
to_x, to_y = ct.transform(to_lon, to_lat)

# 2. Initializing Crawler
crawler = RealtimeCrawling(CHROM_DRIVER_PATH, WAIT_TIMEOUT, DELAY_RANGE)

# 3. 크롤링
records = [] # 소요시간 리스트

for fx, fy, tx, ty in zip(from_x, from_y, to_x, to_y):

    # 실시간(최적) 대중교통
    try:
        time_transit = crawler.get_transit_time(fx, fy, tx, ty)        
        time.sleep(random.uniform(*DELAY_RANGE))        
    except Exception as e:
        time_transit = None
        print(e)

    # 실시간(추천) 자동차
    try:
        time_car = crawler.get_car_time(fx, fy, tx, ty)
        time.sleep(random.uniform(*DELAY_RANGE))
    except Exception as e:
        time_car = None
        print(e)

    print('대중교통 (최적)소요시간:', time_transit)
    print('자동차 (실시간추천)소요시간:', time_car)
    print('\n')

    records.append({
        'from_x': fx, 'from_y': fy,
        'to_x': tx, 'to_y': ty,
        'time_transit': time_transit,
        'time_car': time_car
    })
    time.sleep(random.uniform(*DELAY_RANGE))

crawler.close() # 크롤링 종료

# 4. Print and Save data frame
df_result = pd.DataFrame(records)
print(df_result)

df_result.to_csv('df_realtime.csv', index = False, na_rep = 'None', encoding = 'cp949')