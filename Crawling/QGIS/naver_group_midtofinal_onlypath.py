import pandas as pd
import requests
from datetime import datetime, timedelta
import time

# 네이버 API 키
NAVER_CLIENT_ID = "oa77i9oz1h"
NAVER_CLIENT_SECRET = "xvC9h8wAZLXjsokASCSKjLfNJ5uR63sKBGz705KA"

# 날짜

# # 범위
# START_ROW = 0
# END_ROW = 1000

def parse_datetime(time_str):
    try:
        days = int(time_str.split("일")[0].replace("0+", ""))
        hour = int(time_str.split("일")[1].replace("시", "").strip())
        target = datetime.now() + timedelta(days=days)
        return target.replace(hour=hour, minute=0, second=0, microsecond=0)
    except Exception:
        return None

def call_naver_api(start_lat, start_lng, end_lat, end_lng):
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }
    url = (
        "https://maps.apigw.ntruss.com/map-direction/v1/driving"
        f"?start={start_lng},{start_lat}&goal={end_lng},{end_lat}&option=trafast"
    )
    res = requests.get(url, headers=headers, timeout=20)
    res.raise_for_status()
    return res.json()

def run_simulation(day, hour):
    df = pd.read_csv(fr".\datasets\df_path\df_path_{day}_{hour}.csv", encoding = 'cp949')#.iloc[START_ROW:END_ROW]

    # num -> [ "lon,lat", "lon,lat", ... ] 형태로 저장
    paths_by_num = {}

    for _, row in df.iterrows():
        num = int(row["idx"])
        start_lat, start_lng = row["mid_lat"], row["mid_lng"]
        end_lat, end_lng = row["end_lat"], row["end_lng"]

        try:

            data = call_naver_api(start_lat, start_lng, end_lat, end_lng)
            if "route" not in data or not data["route"].get("trafast"):
                print(f"🛑 {num}: 네이버 API 결과 없음")
                paths_by_num[num] = []
                continue

            # path: [[x,y], [x,y], ...]  => "x,y" 문자열로 변환해 리스트로 저장
            path_xy = data["route"]["trafast"][0]["path"]
            paths_by_num[num] = [f"{x},{y}" for x, y in path_xy]

            print(f"✅ {num}: 경로 포인트 {len(paths_by_num[num])}개")

        except Exception as e:
            print(f"🛑 {num} 처리 실패: {e}")
            paths_by_num[num] = []

        time.sleep(0.2)

    # 열 = num, 행 = 각 좌표(길이가 가장 긴 경로 기준으로 행 수 맞춤)
    nums_sorted = sorted(paths_by_num.keys())
    max_len = max((len(v) for v in paths_by_num.values()), default=0)

    rows = []
    for i in range(max_len):
        row_dict = {}
        for n in nums_sorted:
            coords = paths_by_num[n]
            row_dict[str(n)] = coords[i] if i < len(coords) else ""
        rows.append(row_dict)

    out_df = pd.DataFrame(rows, columns=[str(n) for n in nums_sorted])
    # out_name = f"naver_paths_matrix_{day}_{START_ROW}_{END_ROW}.csv"
    out_name = fr".\results\paths_matrix\naver_paths_matrix_{day}_{hour}.csv"
    out_df.to_csv(out_name, index=False, encoding="utf-8-sig")
    print("📁 저장 완료:", out_name)

if __name__ == "__main__":
    for day in range(1, 31):
        for hour in range(0, 23):

            try:
                run_simulation(day=day, hour=hour)
            except FileNotFoundError:
                print(f"⚠️ 파일 없음: geo_group_midtofinal_{day}_{hour}.csv → 건너뜀")
                continue