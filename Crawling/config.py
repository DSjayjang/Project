"""
CHROM_DRIVER_PATH: 크롬 드라이버가 설치된 내 PC 경로
WAIT_TIMEOUT: 최대 몇 초까지 로딩을 기다릴 것인지
DELAY_RANGE: 딜레이 시간 (min, max)
"""
CHROM_DRIVER_PATH = r'C:\Users\user\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe'
WAIT_TIMEOUT = 10
DELAY_RANGE = (1.0, 2.0)

"""
■ CSS Selector
SELECTOR_TRANSIT: 대중교통
SELECTOR_CAR: 자동차

TIME_OPTION_BTN: 시간 옵션
CALENDAR_BTN: 캘린더 열기

PRV_BTN: 이전 달 버튼
NXT_BTN: 다음 달 버튼
CUR_TXT: 현재 년월

XPATH_DAY: 일자 선택
"""
SELECTOR_TRANSIT = '#tab_pubtrans_directions > ul > li.sc-1it5x3x.ckQsSP.is_selected > div > div > div > em'
SELECTOR_CAR = '#section_content > div > div.sc-hz6zmc.eOGmI > div.sc-1nqk12w.kHPMtO > div.direction_summary_list_wrap > ul > li:nth-child(1) > div > div > div.route_summary_info_duration > strong'

TIME_OPTION_BTN = '#section_content > div > div.sc-hz6zmc.eOGmI > div > div.sc-dzen53.hBBcri > div > button.btn_time_option.btn_option'
CALENDAR_BTN = '#section_content > div > div.sc-hz6zmc.eOGmI > div > div.sc-dzen53.hBBcri > div.sc-7nk3kt.bsQxOy > div.timeset_option.timeset_option_day > div > button'

PRV_BTN = '#section_content > div > div.sc-hz6zmc.eOGmI > div > div.sc-dzen53.hBBcri > div.sc-7nk3kt.bsQxOy > div.timeset_option.timeset_option_day > div > div > div.nav_calendar > button.btn_prv_month'
NXT_BTN = '#section_content > div > div.sc-hz6zmc.eOGmI > div > div.sc-dzen53.hBBcri > div.sc-7nk3kt.bsQxOy > div.timeset_option.timeset_option_day > div > div > div.nav_calendar > button.btn_nxt_month'
CUR_TXT = '#section_content > div > div.sc-hz6zmc.eOGmI > div > div.sc-dzen53.hBBcri > div.sc-7nk3kt.bsQxOy > div.timeset_option.timeset_option_day > div > div > div.nav_calendar > span'

XPATH_DAY = "//*[@id='section_content']//div[contains(@class,'table_calendar')]//button[normalize-space(text())='{day}']"
