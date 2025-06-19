import time
import random

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import config

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
        elem = self.wait.until(EC.presence_of_element_located((
            By.CSS_SELECTOR, selector)))
        time.sleep(random.uniform(*self.delay_range))

        return elem.text

    def get_transit_time(self, fx, fy, tx, ty):
        url = f'https://map.naver.com/p/directions/{fx},{fy}/{tx},{ty}/-/transit?c=14.00,0,0,0,dh'
        return self._fetch(url, config.SELECTOR_TRANSIT)

    def get_car_time(self, fx, fy, tx, ty):
        url = f'https://map.naver.com/p/directions/{fx},{fy}/{tx},{ty}/-/car?c=14.00,0,0,0,dh'
        return self._fetch(url, config.SELECTOR_CAR)
    
    def close(self):
        self.driver.quit()
