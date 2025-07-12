import os
import time
import random
import pandas as pd

from config import CHROM_DRIVER_PATH, WAIT_TIMEOUT, DELAY_RANGE
from utils import CoordinateTransformer, DatetimeValidator
from crawler import SpecificCrawling

# Data Load
os.chdir(r'C:\Users\user\Desktop\연구\5. 국방부 용역과제') # pcrl
# os.chdir(r'C:\Users\linde\OneDrive\Desktop\3. 연구\5. 국방부 용역과제') # my pc
# os.chdir(r'C:\Users\linde\Desktop\연구\5. 국방부 용역') # laptop
df = pd.read_csv('DB.csv')

"""
# 조회할 시간대 선택

Args:
dt_year (int): 연도
dt_month (int, 1-12): 월
dt_day (int, 1-31): 일
dt_hour (int, 0-23): 시간
dt_min (int, {0, 10, 20, 30, 40, 50}): 분
"""
dt_year = 2025
dt_month = 6
dt_day = 25
dt_hour = 23
dt_minute = 20


# 시간 유효성 검사
try:
    DatetimeValidator.validator(dt_year, dt_month, dt_day, dt_hour, dt_minute)
except ValueError as e:
    print('입력 오류:', e)
    raise

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
crawler = SpecificCrawling(CHROM_DRIVER_PATH, WAIT_TIMEOUT, DELAY_RANGE)

# 3. 크롤링
records = [] # 소요시간 리스트

for fx, fy, tx, ty in zip(from_x, from_y, to_x, to_y):
    try:
        time_transit_specific = crawler.choose_time(fx, fy, tx, ty, dt_year, dt_month, dt_day, dt_hour, dt_minute)
        time.sleep(random.uniform(*DELAY_RANGE))

    except Exception as e:
        time_transit_specific = None
        import traceback; traceback.print_exc()
        print(e)
    
    print(f'대중교통 {dt_year}/{dt_month}/{dt_day} {dt_hour:02d}:{dt_minute} 출발 시 소요시간:', time_transit_specific)
    print('\n')

    records.append({
        'from_x': fx, 'from_y': fy,
        'to_x': tx, 'to_y': ty,
        'time_transit_specific': time_transit_specific,
    })
    time.sleep(random.uniform(*DELAY_RANGE))

crawler.close() # 크롤링 종료

# 4. Print and Save data frame
df_result = pd.DataFrame(records)
print(df_result)

df_result.to_csv('df_specific.csv', index = False, na_rep = 'None', encoding = 'cp949')