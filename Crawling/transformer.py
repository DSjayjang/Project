"""
■ 좌표변환
위/경도 > 네이버지도 좌표계로 변환
"""

from pyproj import Transformer

class CoordinateTransformer:
    def __init__(self, from_crs = "EPSG:4326", to_crs = "EPSG:3857"):
        self._tf = Transformer.from_crs(from_crs, to_crs, always_xy = True)
        
    def transform(self, longitude: float, latitude: float):
        """
        longitude: 1D numpy ndarray
        latitude: 1D numpy ndarray
        """        
        return self._tf.transform(longitude, latitude)