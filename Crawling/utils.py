from pyproj import Transformer
import calendar
from typing import ClassVar, Set

class CoordinateTransformer:
    """
    좌표변환 클래스
    위/경도 > 네이버지도 좌표계로 변환
    """
    def __init__(self, from_crs = "EPSG:4326", to_crs = "EPSG:3857"):
        self._tf = Transformer.from_crs(from_crs, to_crs, always_xy = True)
        
    def transform(self, longitude: float, latitude: float):
        """
        params:
        longitude: 1D numpy ndarray
        latitude: 1D numpy ndarray

        returns:
        tuple of 1D numpy ndarray
        (to_x, to_y) 변환된 x, y 좌표
        """     
        return self._tf.transform(longitude, latitude)

class DatetimeValidator:
    """
    날짜 유효성 검사 클래스
    """
    ALLOWED_MINUTES: ClassVar[Set[int]] = {0, 10, 20, 30, 40, 50}

    @classmethod
    def validator(cls,
                  year: int, month: int, day: int,
                  hour:int, minute: int) -> None:
        # 월(month) 검사
        if not 1 <= month <= 12:
            raise ValueError(f'잘못된 월 입력: month={month}')

        # 일(day) 검사
        max_day = calendar.monthrange(year, month)[1]
        if not 1 <= day <= max_day:
            raise ValueError(f'잘못된 일 입력: day={day} (1~{max_day} 사이로 입력)')
        
        # 시(hour) 검사
        if not 0 <= hour <= 23:
            raise ValueError(f'잘못된 시간 입력: hour={hour} (0~23 사이로 입력)')
        
        # 분(min) 검사
        if minute not in cls.ALLOWED_MINUTES:
            raise ValueError(f'잘못된 분 입력: minute={minute} ({cls.ALLOWED_MINUTES} 중 하나 입력)')