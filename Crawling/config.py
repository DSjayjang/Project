"""
CHROM_DRIVER_PATH: 크롬 드라이버가 설치된 내 PC 경로
WAIT_TIMEOUT: 최대 몇 초까지 로딩을 기다릴 것인지
DELAY_RANGE: 딜레이 시간 (min, max)
"""

CHROM_DRIVER_PATH = r'C:\Users\user\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe'
WAIT_TIMEOUT = 10
DELAY_RANGE = (1.0, 2.0)

# CSS Selector
SELECTOR_TRANSIT = '#tab_pubtrans_directions > ul > li.sc-1it5x3x.ckQsSP.is_selected > div > div > div > em'
SELECTOR_CAR = '#section_content > div > div.sc-hz6zmc.eOGmI > div.sc-1nqk12w.kHPMtO > div.direction_summary_list_wrap > ul > li:nth-child(1) > div > div > div.route_summary_info_duration > strong'