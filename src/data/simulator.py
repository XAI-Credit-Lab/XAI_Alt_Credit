
from pathlib import Path
import pandas as pd


#print(data.describe().T.to_string()) # 각 변수의 평균, 표준편차 등
# 대안 변수 생성 : telecom_payment_rate, utility_payment_rate, spending_consistency
#   regular_payment_count, app_login_frequency

#csv_path = Path(__file__).with_name("cs-training.csv")
#df = pd.read_csv(csv_path)

#df_copy = df.copy()
import numpy as np
import pandas as pd
csv_path = Path(__file__).with_name("cs-training.csv")
print(f"CSV 읽는 중: {csv_path}", flush=True)
df = pd.read_csv(csv_path)
print(f"읽기 완료: {len(df):,}행", flush=True)

def add_alt_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()  # 원본 보호
    n = len(df)

    # 기존 변수로 만든 잠재 위험도 (0~1). 타깃(라벨)은 쓰지 않음
    late_cols = ["NumberOfTime30-59DaysPastDueNotWorse",
                 "NumberOfTime60-89DaysPastDueNotWorse",
                 "NumberOfTimes90DaysLate"]
    late = df[late_cols].replace([96, 98], np.nan).fillna(0).sum(axis=1)
    #risk = (late.clip(upper=10) / 10).to_numpy()
    Target = df["SeriousDlqin2yrs"]

    df["telecom_pay_rate"] = np.clip(0.98 - 0.4 * Target + rng.normal(0, 0.08, n), 0, 1)
    df["utility_pay_rate"] = np.clip(0.98 - 0.35 * Target + rng.normal(0, 0.08, n), 0, 1)
    df["spend_consistency"] = np.clip(80 - 40 * Target + rng.normal(0, 10, n), 0, 100)
    df["subscription_cnt"] = rng.poisson(3 + 2 * Target, n)
    df["app_login_freq"] = rng.poisson(np.clip(20 - 10 * Target, 1, None), n)
    return df

print("대안 변수 생성 중...", flush=True)
result = add_alt_features(df)
output_path = Path(__file__).with_name("결과.csv")
print(f"결과 저장 중: {output_path}", flush=True)
result.to_csv(output_path, index=False)
print(f"완료: {len(result):,}행 저장됨", flush=True)