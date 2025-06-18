# from user_agent import generate_user_agent, generate_navigator
# user_agent = generate_user_agent()
# user_agent

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

# 셋업
chrome_path = r'C:\Users\user\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe'
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')
service = Service(executable_path = chrome_path)
driver = webdriver.Chrome(service = service, options = options)

# # 네이버지도 길찾기 열기
# url = "https://map.naver.com/p/directions/-/-/-/transit?c=11.00,0,0,0,dh"

url = " `"


driver.get(url)


# 소요 시간 요소 찾기
try:
    time.sleep(2)
    time_element = driver.find_element(
        By.CSS_SELECTOR,
        "#tab_pubtrans_directions > ul > li.sc-1it5x3x.ckQsSP.is_selected > div > div > div > em"
    )
    print("[RESULT] 예상 소요 시간:", time_element.text)
except Exception as e:
    print("[ERROR] 소요 시간 추출 실패:", e)

# 종료 대기
input("Enter를 누르면 브라우저를 닫습니다.")
driver.quit()