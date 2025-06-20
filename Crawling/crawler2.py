import time
import random

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import config
from config import TIME_OPTION_BTN, CALENDAR_BTN, PRV_BTN, NXT_BTN, CUR_TXT, XPATH_DAY

class NaverMapCrawler:
    def __init__(self, driver_path, wait_timeout, delay_range):
        options = webdriver.ChromeOptions()
        options.add_argument('--start-maximized')
        service = Service(executable_path = driver_path)

        self.driver = webdriver.Chrome(service = service, options = options)
        self.wait = WebDriverWait(self.driver, wait_timeout)
        self.delay_range = delay_range

    def _fetch(self, url, selector):
        self.driver.get(url)
        elem = self.wait.until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                selector)))
        time.sleep(random.uniform(*self.delay_range)) # 대기

        return elem.text

    # 1. 실시간 대중교통
    def get_transit_time(self, fx: list, fy: list, tx: list, ty: list) -> str:
        url = f'https://map.naver.com/p/directions/{fx},{fy}/{tx},{ty}/-/transit?c=14.00,0,0,0,dh'

        return self._fetch(url, config.SELECTOR_TRANSIT)

    # 2. 실시간 자동차
    def get_car_time(self, fx: list, fy: list, tx: list, ty: list) -> str:
        url = f'https://map.naver.com/p/directions/{fx},{fy}/{tx},{ty}/-/car?c=14.00,0,0,0,dh'

        return self._fetch(url, config.SELECTOR_CAR)
    
    # 3. 캘린더 열기
    def open_calendar(self, fx, fy, tx, ty, target_year, target_month, target_day):
        # 1. 길찾기 페이지 열기
        url = f'https://map.naver.com/p/directions/{fx},{fy}/{tx},{ty}/-/transit?c=14.00,0,0,0,dh'
        self.driver.get(url)

        # 2. 시간 옵션 열기
        time_btn = self.wait.until(
            EC.element_to_be_clickable((
                By.CSS_SELECTOR, TIME_OPTION_BTN)))
        time_btn.click()
        time.sleep(random.uniform(*self.delay_range))

        # 3. 캘린더 버튼 클릭
        btn = self.wait.until(
            EC.element_to_be_clickable((
                By.CSS_SELECTOR, CALENDAR_BTN)))
        btn.click()
        time.sleep(random.uniform(*self.delay_range))

        # 4. 년/월 선택
        if target_year and target_month:
            while True:
                cur = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, CUR_TXT))).text.strip()
                # 예: '2025년 6월'
                y_str, m_str = cur.replace('월', '').split('년')
                cur_y = int(y_str)
                cur_m = int(m_str)

                if (cur_y > target_year) or (cur_y == target_year and cur_m > target_month):
                    self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, PRV_BTN))).click()
                elif (cur_y < target_year) or (cur_y == target_year and cur_m < target_month):
                    self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, NXT_BTN))).click()
                else:
                    break
                time.sleep(random.uniform(*self.delay_range))
        
        # 5. 날짜 선택
        xpath = XPATH_DAY.format(day = target_day)
        day_buttons = self.driver.find_elements(By.XPATH, xpath)
        if day_buttons:
            btn = day_buttons[0]
            self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
            btn.click()
        else:
            raise Exception('에러: 날짜를 찾을 수 없음')

        time.sleep(random.uniform(*self.delay_range))

        # 6. 시간 선택


    # Selenium 종료
    def close(self):
        self.driver.quit()