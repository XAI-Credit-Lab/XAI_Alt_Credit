import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__) #로그 생성
# __name__ = 현재 모듈의 이름을 담고 있는 파이썬 내장 변수

def load_gmsc(
    path: Union[str, Path] = Path(__file__).resolve().with_name("cs-training.csv"),
) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_ratio(%)": (df.isna().mean() * 100).round(2),
    })
    logger.info("shape: %s\n%s", df.shape, summary.to_string())
    return df