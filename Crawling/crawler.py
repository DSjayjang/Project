import time
import random

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from config import SELECTOR_TRANSIT, SELECTOR_TRANSIT_SPECIFIC, SELECTOR_CAR, TIME_OPTION_BTN
from config import CALENDAR_BTN, PRV_BTN, NXT_BTN, CUR_TXT, XPATH_DAY
from config import HOUR_BTN, XPATH_HOUR, MIN_BTN, XPATH_MIN

class NaverMapCrawler:
    def __init__(self, driver_path, wait_timeout, delay_range):
        options = webdriver.ChromeOptions()
        options.add_argument('--start-maximized')
        service = Service(executable_path = driver_path)

        self.driver = webdriver.Chrome(service = service, options = options)
        self.wait = WebDriverWait(self.driver, wait_timeout)
        self.delay_range = delay_range

    # 소요시간 반환
    def _fetch(self, url, selector):
        self.driver.get(url)
        elem = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        time.sleep(random.uniform(*self.delay_range))

        return elem.text

    # Selenium 종료
    def close(self):
        self.driver.quit()


class RealtimeCrawling(NaverMapCrawler):
    # 실시간 대중교통
    def get_transit_time(self, fx: list, fy: list, tx: list, ty: list) -> str:
        url = f'https://map.naver.com/p/directions/{fx},{fy}/{tx},{ty}/-/transit?c=14.00,0,0,0,dh'

        return super()._fetch(url, SELECTOR_TRANSIT)
    
    # 실시간 자동차
    def get_car_time(self, fx:list, fy:list, tx:list, ty:list) -> str:
        url = f'https://map.naver.com/p/directions/{fx},{fy}/{tx},{ty}/-/car?c=14.00,0,0,0,dh'

        return super()._fetch(url, SELECTOR_CAR)


class SpecificCrawling(NaverMapCrawler):
        # Method Overriding
        def _fetch(self, selector):
            elem = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            time.sleep(random.uniform(*self.delay_range))

            return elem.text

        def choose_time(self, fx:list, fy:list, tx:list, ty:list, 
                      dt_year:int, dt_month:int, dt_day:int, 
                      dt_hour:int, dt_min:int) -> str:
            # 1. 길찾기 페이지 열기
            url = f'https://map.naver.com/p/directions/{fx},{fy}/{tx},{ty}/-/transit?c=14.00,0,0,0,dh'
            self.driver.get(url)

            # 2. 시간 옵션 열기
            time_btn = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, TIME_OPTION_BTN)))
            time_btn.click()
            time.sleep(random.uniform(*self.delay_range))

            # 3-1. 캘린더 버튼 클릭
            cal_btn = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, CALENDAR_BTN)))
            cal_btn.click()
            time.sleep(random.uniform(*self.delay_range))

            # 3-2. 년/월 선택
            if dt_year and dt_month:
                while True:
                    cur = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, CUR_TXT))).text.strip()
                    # 예: '2025년 6월'
                    y_str, m_str = cur.replace('월', '').split('년')
                    cur_y = int(y_str)
                    cur_m = int(m_str)

                    if (cur_y > dt_year) or (cur_y == dt_year and cur_m > dt_month):
                        self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, PRV_BTN))).click()
                    elif (cur_y < dt_year) or (cur_y == dt_year and cur_m < dt_month):
                        self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, NXT_BTN))).click()
                    else:
                        break
                    time.sleep(random.uniform(*self.delay_range))
            
            # 3-3. 날짜 선택
            xpath_day = XPATH_DAY.format(day = dt_day)
            day_buttons = self.driver.find_elements(By.XPATH, xpath_day)[0]
            self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath_day)))
            day_buttons.click()
            time.sleep(random.uniform(*self.delay_range))

            # 4-1. 시간 버튼 클릭
            hour_btn = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, HOUR_BTN)))
            hour_btn.click()
            time.sleep(random.uniform(*self.delay_range))

            # 4-2 XPATH로 dt_hour 버튼 찾기
            xpath_hour = XPATH_HOUR.format(hour = dt_hour)
            hour_bottns = self.driver.find_element(By.XPATH, xpath_hour)

            # 4-3 스크롤하여 화면 중앙에 보이게 한 뒤 클릭
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", hour_bottns)
            self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath_hour))).click()
            time.sleep(random.uniform(*self.delay_range))

            # 5-1. 분 버튼 클릭
            min_btn = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, MIN_BTN)))
            min_btn.click()
            time.sleep(random.uniform(*self.delay_range))

            # 5-2 XPATH로 target_min 버튼 찾기
            xpath_min = XPATH_MIN.format(min = dt_min)
            min_bottns = self.driver.find_element(By.XPATH, xpath_min)

            # 5-3 스크롤하여 화면 중앙에 보이게 한 뒤 클릭
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", min_bottns)
            self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath_min))).click()
            time.sleep(random.uniform(*self.delay_range))

            return self._fetch(SELECTOR_TRANSIT_SPECIFIC)
