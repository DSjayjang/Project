import os
import time
import random
import pandas as pd

from config import CHROM_DRIVER_PATH, WAIT_TIMEOUT, DELAY_RANGE
from transformer import CoordinateTransformer
from crawler import NaverMapCrawler

os.chdir(r'C:\Users\user\Desktop\연구\5. 국방부 용역과제')
df = pd.read_csv('DB.csv')

# Data Load

# 출발지/목적지 위/경도
"""
from_lat, from_lon: 출발지 위도, 경도
to_lat, to_lon: 목적지 위도, 경도 >> ndarray로 변경필요
"""
from_lat = df['Latitude'][:3].to_numpy()
from_lon = df['Longitude'][:3].to_numpy()
to_lat = 37.450141
to_lon = 126.653467


# 1. 좌표 변환
ct = CoordinateTransformer()
from_x, from_y = ct.transform(from_lon, from_lat)
to_x, to_y = ct.transform(to_lon, to_lat)

# 2. Initializing Crawler
crawler = NaverMapCrawler(CHROM_DRIVER_PATH, WAIT_TIMEOUT, DELAY_RANGE)

# 3. 소요시간 계산
records = []

for fx, fy in zip(from_x, from_y):
    try:
        # 1. 대중교통
        time_transit = crawler.get_transit_time(fx, fy, to_x, to_y)
        
        time.sleep(random.uniform(*DELAY_RANGE))
        
        # 2. 자동차
        time_car = crawler.get_car_time(fx, fy, to_x, to_y)

    except Exception as e:
        time_transit = None
        time_car = None
        print(e)
    
    print("대중교통 (최적)소요시간:", time_transit)
    print("자동차 (실시간추천)소요시간:", time_car)

    records.append({
        'from_x': fx, 'from_y': fy,
        'to_x': to_x, 'to_y': to_y,
        'time_transit': time_transit,
        'time_car': time_car
    })

    time.sleep(random.uniform(*DELAY_RANGE))

# 4. 종료
crawler.close()

df_result = pd.DataFrame(records)
print(df_result)

