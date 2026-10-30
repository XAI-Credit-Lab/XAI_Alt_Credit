"""Give Me Some Credit 데이터셋 로더."""
import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent   # src/data


def _resolve_path(path: Optional[Union[str, Path]]) -> Path:
    if path is not None:
        return Path(path)
    data_dir = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
    return data_dir / os.getenv("GMSC_FILENAME", "cs-training.csv")


def load_gmsc(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """GMSC 학습 데이터를 로드하고 컬럼명·dtype·결측 비율을 로깅한다."""
    path = _resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 다음 명령으로 다운로드하세요:\n"
            "  kaggle competitions download -c GiveMeSomeCredit"
        )

    df = pd.read_csv(path, index_col=0)

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_ratio(%)": (df.isna().mean() * 100).round(2),
    })
    logger.info("Loaded %s | shape=%s\n%s", path.name, df.shape, summary.to_string())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    load_gmsc()