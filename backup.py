import os
import time
import random
import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from pyproj import Transformer # 좌표변환

# Data Load
os.chdir(r'C:\Users\user\Desktop\연구\5. 국방부 용역과제')
df = pd.read_csv('DB.csv')

# 출발지/목적지 위/경도
"""
from_lat, from_lon: 출발지 위도, 경도
to_lat, to_lon: 목적지 위도, 경도 >> ndarray로 변경필요
"""
from_lat = df['Latitude'][:20].to_numpy()
from_lon = df['Longitude'][:20].to_numpy()
to_lat = 37.450141
to_lon = 126.653467

# 좌표변환: WGS84 -> Web Mercator
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy = True)
from_x, from_y = transformer.transform(from_lon, from_lat)
to_x, to_y = transformer.transform(to_lon, to_lat)

# url 리스트 생성
"""
리스트 각 원소는 출발지 -> 목적지까지의 네이버지도 경로

urls_transit: 대중교통
urls_transit: 자동차
"""
urls_transit = [f'https://map.naver.com/p/directions/{fx},{fy}/{tx},{ty}/-/transit?c=14.00,0,0,0,dh'
        for fx, fy, tx, ty in zip(from_x, from_y, [to_x]*len(from_x), [to_y]*len(from_x))]
urls_car = [f'https://map.naver.com/p/directions/{fx},{fy}/{tx},{ty}/-/car?c=14.00,0,0,0,dh'
        for fx, fy, tx, ty in zip(from_x, from_y, [to_x]*len(from_x), [to_y]*len(from_x))]

# CSS Selector
selector_transit = '#tab_pubtrans_directions > ul > li.sc-1it5x3x.ckQsSP.is_selected > div > div > div > em'
selector_car = '#section_content > div > div.sc-hz6zmc.eOGmI > div.sc-1nqk12w.kHPMtO > div.direction_summary_list_wrap > ul > li:nth-child(1) > div > div > div.route_summary_info_duration > strong'

# Selenium Setting
chrome_path = r'C:\Users\user\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe'
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')
service = Service(executable_path = chrome_path)
driver = webdriver.Chrome(service = service, options = options)
wait = WebDriverWait(driver, 10) # 최대 10초 대기

time_transit = []
time_car = []

# 시간 계산
for ut, uc in zip(urls_transit, urls_car):
    start = time.time()
    # 1. 대중교통
    driver.get(ut)
    try:
        time_element = wait.until(EC.presence_of_element_located((
            By.CSS_SELECTOR,
            selector_transit)))

        time_transit.append(time_element.text)
        print("대중교통 (최적)소요시간:", time_element.text)

    except Exception as e:
        time_transit.append(None)
    time.sleep(random.uniform(1,2))

    # 2. 자동차
    driver.get(uc)
    try:
        time_element = wait.until(EC.presence_of_element_located((
            By.CSS_SELECTOR,
            selector_car)))

        time_car.append(time_element.text)
        print("자동차 (실시간추천)소요시간:", time_element.text)

    except Exception as e:
        time_transit.append(None)
    time.sleep(random.uniform(1,2))

end = time.time()
diff = end - start

driver.quit()

print('대중교통 (최적)소요시간:', time_transit)
print('자동차 (실시간추천)소요시간:', time_car)

# 데이터프레임으로 저장
df_result = pd.DataFrame({
    'from_x': from_x,
    'from_y': from_y,
    'to_x': to_x,
    'to_y': to_y,
    'time_transit': time_transit,
    'time_car': time_car})
print(df_result)
print('걸린시간:', diff)
