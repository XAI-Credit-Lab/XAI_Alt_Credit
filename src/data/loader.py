
'''
from pathlib import Path
import pandas as pd

csv_path = Path(__file__).with_name("cs-training.csv")
data = pd.read_csv(csv_path)

summary = pd.DataFrame({
    "Variable": data.columns, #변수명
    "Type": data.dtypes.astype(str).values, #문자열로 바꿈
    "Missing": data.isna().sum().values, #결측치 개수
})
summary.index += 1 # 0,1,2 -> 1,2,3
summary.index.name = "No." # 인덱스 이름 No. 로 설정
print(summary.to_string()) # 요약 정보 출력
'''
import logging
import pandas as pd

logger = logging.getLogger(__name__) #로그 생성
# __name__ = 현재 모듈의 이름을 담고 있는 파이썬 내장 변수

def load_gmsc(path: str = "data/raw/cs-training.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_ratio(%)": (df.isna().mean() * 100).round(2),
    })
    logger.info("shape: %s\n%s", df.shape, summary.to_string())
    return df

