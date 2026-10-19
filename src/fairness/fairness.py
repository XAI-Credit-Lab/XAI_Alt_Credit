"""
src/fairness/metrics.py
공정성 지표 모듈 (초안 - 시뮬레이터에서 쓰는 부분만 먼저 작성)

시뮬레이터(src/data/simulator.py)가 import하는 이름:
  AGE_BINS, AGE_LABELS, gender_di, age_di, fmt_age_di
→ 이름과 입력/반환 형태를 바꾸면 시뮬레이터도 같이 수정해야 함
"""
import numpy as np
import pandas as pd

# 보호 속성: 연령대 구간 (시뮬레이터의 add_age_group이 이 값으로 age_group 생성)
AGE_BINS   = [0, 34, 54, 200]
AGE_LABELS = ["~34", "35~54", "55+"]
AGE_REFERENCE = "35~54"   # 연령 DI의 기준(분모) 집단


# 시뮬레이터 리포트와 공정성 검증이 공통으로 사용하는 DI 계산 함수
def gender_di(df, approved):
    """성별 DI = 여성 승인율 / 남성 승인율"""
    approved = np.asarray(approved, dtype=float)
    g = df["gender"].to_numpy()
    return approved[g == 1].mean() / approved[g == 0].mean()


def age_di(df, approved, reference=AGE_REFERENCE):
    """연령대 DI = 각 연령대 승인율 / 기준 연령대(35~54) 승인율 (기준 집단 제외한 dict)"""
    rates = pd.Series(np.asarray(approved, dtype=float), index=df.index).groupby(df["age_group"]).mean()
    return {grp: rates[grp] / rates[reference] for grp in AGE_LABELS if grp != reference}


def fmt_age_di(di_dict):
    """age_di 결과를 출력용 문자열로 변환"""
    return ", ".join(f"{grp}/{AGE_REFERENCE} {v:.3f}" + ("(0.8 미만)" if v < 0.8 else "") for grp, v in di_dict.items())
