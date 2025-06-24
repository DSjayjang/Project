"""
CHROM_DRIVER_PATH: 크롬 드라이버가 설치된 경로
WAIT_TIMEOUT: 기다릴 최대 로딩시간(초)
DELAY_RANGE: 클릭 딜레이 시간(초) (min, max)
"""
CHROM_DRIVER_PATH = r'C:\Users\user\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe' # pcrl
# CHROM_DRIVER_PATH = r'C:\Users\linde\OneDrive\Desktop\3. 연구\5. 국방부 용역과제\chromedriver-win64\chromedriver-win64\chromedriver.exe' # my pc
# CHROM_DRIVER_PATH = r'C:\Users\linde\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe' # laptop
WAIT_TIMEOUT = 10
DELAY_RANGE = (1.0, 2.0)

"""
■ CSS Selector
SELECTOR_TRANSIT: 대중교통 최적시간
SELECTOR_TRANSIT_SPECIFIC: 대중교통 특정시간
SELECTOR_CAR: 자동차 실시간
TIME_OPTION_BTN: 시간 옵션 열기

CALENDAR_BTN: 캘린더 열기
PRV_BTN: 이전 달 버튼
NXT_BTN: 다음 달 버튼
CUR_TXT: 현재 년월
XPATH_DAY: 일자 선택

HOUR_BTN: 시간(hour) 열기
XPATH_HOUR: 시간(hour) 선택
MIN_BTN: 분(minutes) 열기
XPATH_MIN: 분(minutes) 선택
"""
SELECTOR_TRANSIT = '#tab_pubtrans_directions > ul > li.sc-1it5x3x.ckQsSP.is_selected > div > div > div > em'
SELECTOR_TRANSIT_SPECIFIC = '#tab_pubtrans_directions > ul > li > div > div > div > em'
SELECTOR_CAR = '#section_content > div > div.sc-hz6zmc.eOGmI > div.sc-1nqk12w.kHPMtO > div.direction_summary_list_wrap > ul > li:nth-child(1) > div > div > div.route_summary_info_duration > strong'
TIME_OPTION_BTN = '#section_content > div > div.sc-hz6zmc.eOGmI > div > div.sc-dzen53.hBBcri > div > button.btn_time_option.btn_option'

# 년/월 CSS Selector/Xpath
CALENDAR_BTN = '#section_content > div > div.sc-hz6zmc.eOGmI > div > div.sc-dzen53.hBBcri > div.sc-7nk3kt.bsQxOy > div.timeset_option.timeset_option_day > div > button'
PRV_BTN = '#section_content > div > div.sc-hz6zmc.eOGmI > div > div.sc-dzen53.hBBcri > div.sc-7nk3kt.bsQxOy > div.timeset_option.timeset_option_day > div > div > div.nav_calendar > button.btn_prv_month'
NXT_BTN = '#section_content > div > div.sc-hz6zmc.eOGmI > div > div.sc-dzen53.hBBcri > div.sc-7nk3kt.bsQxOy > div.timeset_option.timeset_option_day > div > div > div.nav_calendar > button.btn_nxt_month'
CUR_TXT = '#section_content > div > div.sc-hz6zmc.eOGmI > div > div.sc-dzen53.hBBcri > div.sc-7nk3kt.bsQxOy > div.timeset_option.timeset_option_day > div > div > div.nav_calendar > span'
XPATH_DAY = "//*[@id='section_content']//div[contains(@class,'table_calendar')]//button[normalize-space(text())='{day}']"

# 시간(hour) CSS Selector/Xpath
HOUR_BTN = '#section_content > div > div.sc-hz6zmc.eOGmI > div > div.sc-dzen53.hBBcri > div.sc-7nk3kt.bsQxOy > div.timeset_option.timeset_option_hour > div > button'
XPATH_HOUR = ".//li/button[normalize-space(text())='{hour:02d}']"

# 분(minute) CSS Selector/Xpath
MIN_BTN = '#section_content > div > div.sc-hz6zmc.eOGmI > div > div.sc-dzen53.hBBcri > div.sc-7nk3kt.bsQxOy > div.timeset_option.timeset_option_minute > div > button'
XPATH_MIN = ".//li/button[normalize-space(text())='{min:02d}']"
