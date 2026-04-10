import time
import numpy as np
import pandas as pd
import requests

df = pd.read_excel(r'..\datasets\250716.xlsx')

num = np.array(df['순번'])
start = np.array(df['출발지 주소'])
mid = np.array(df['중간집결지 주소'])
end = np.array(df['최종 도착지 주소'])

def geocoding(address: list):
    url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    
    headers = {
        "x-ncp-apigw-api-key-id": "클라이언트 아이디",
        "x-ncp-apigw-api-key": "시크릿 키",
        "Accept": "application/json"
    }

    params = {"query": address}
    resp = requests.get(url, headers = headers, params = params)
    resp.raise_for_status()
    data = resp.json()

    try:
        # if data.get('addresses'):
        lat = float(data['addresses'][0]['y']) # 위도
        lng = float(data['addresses'][0]['x']) # 경도
        return lat, lng
    
    except Exception as e:
        print(f'{address} 변환 실패: {e}')

        lat = None
        lng = None
        return lat, lng

# geocoding
if __name__ == "__main__":
    chunk = 1000
    start_index = 0
    last_index = 10
 
    for i in range(start_index, last_index, chunk):
        slc = slice(i, min(i + chunk, last_index))
        print(f'Processing records {slc.start} to {slc.stop}...')

        t0 = time.time()
        start_lat, start_lng = zip(*(geocoding(addr) for addr in start[slc]))
        mid_lat, mid_lng = zip(*(geocoding(addr) for addr in mid[slc]))
        end_lat, end_lng = zip(*(geocoding(addr) for addr in end[slc]))
        t1 = time.time()

        print(f'걸린시간(분): {(t1-t0)/60:.2f}')

        df_chunk = pd.DataFrame({'num': num[slc],
                       'start_lat': start_lat,
                       'start_lng': start_lng,
                       'mid_lat': mid_lat,
                       'mid_lng': mid_lng,
                       'end_lat': end_lat,
                       'end_lng': end_lng})
        
        mode = 'w' if i == start_index else 'a'
        header = True if i == start_index else False

        df_chunk.to_csv(r'..\results\geocoded.csv', mode = mode, header = header, index = False)
        print(f'{slc.start} to {slc.stop} 저장 완료')