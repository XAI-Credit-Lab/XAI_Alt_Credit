"""
src/data/check_debtratio.py
소득이 있는(>1) 행인데 DebtRatio가 비정상적으로 큰 이유 확인 (확인만, 데이터 수정 없음)

확인할 것
  A) 기준값별로 몇 행이 걸리나        -> "DebtRatio가 ○ 초과면 결측" 기준을 정하는 근거
  B) 소득이 작아서 생긴 건가          -> "소득 기준을 1보다 올릴지" 정하는 근거
  C) 시뮬레이터가 만든 건가           -> 씬파일러 값 교체(inject_thin_filers)로 소득과 DebtRatio 짝이 깨졌는지

실행 (프로젝트 루트, .venv 켠 상태):
  python -m src.data.check_debtratio > reports/check_debtratio.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "src" / "data" / "cs-training.csv"
SIM = ROOT / "data" / "processed" / "결과_시나리오.csv"
TARGET, INC, DR = "SeriousDlqin2yrs", "MonthlyIncome", "DebtRatio"
THRESHOLDS = [1, 2, 5, 10, 20, 50, 100, 1000]

pd.set_option("display.width", 200)


def title(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


raw = pd.read_csv(RAW, index_col=0)
raw = raw[raw["age"] > 0].reset_index(drop=True)   # simulator.load_base()와 같게 (나이 0 행 제거)
sim = pd.read_csv(SIM)
aligned = len(raw) == len(sim)                       # 시뮬레이터는 행 순서를 유지하므로 같은 위치 = 같은 사람

# ----------------------------------------------------------------------
title("A) 소득 > 1 인 행: DebtRatio 기준값별로 걸리는 행 수 (원본 vs 시뮬레이터)")
rows = []
for k in THRESHOLDS:
    r = {"기준(초과)": k}
    for name, df in [("원본", raw), ("시뮬레이터", sim)]:
        d = df[df[INC] > 1]
        hit = d[DR] > k
        r[f"{name} 행 수"] = int(hit.sum())
        r[f"{name} 부도율"] = round(d.loc[hit, TARGET].mean(), 4) if hit.any() else np.nan
    rows.append(r)
print(pd.DataFrame(rows).set_index("기준(초과)").to_string())
base = sim.loc[sim[INC] > 1, TARGET].mean()
print(f"  (참고) 시뮬레이터 소득>1 전체 부도율: {base:.4f}")
print("해석")
print("  - 기준을 올려도 행 수가 거의 안 줄어드는 구간이 있으면, 그 위쪽은 대부분 '비율이 아닌 값'")
print("  - 원본보다 시뮬레이터 행 수가 많으면, 늘어난 만큼은 시뮬레이터가 만든 것 (C에서 확인)")

# ----------------------------------------------------------------------
title("B) 소득이 작아서 생긴 건가: 소득 구간별 DebtRatio (원본, 소득 > 1)")
d = raw[raw[INC] > 1]
bins = [1, 10, 100, 500, 1000, 2000, 5000, np.inf]
g = pd.cut(d[INC], bins)
t = d.groupby(g, observed=True)[DR].agg(행수="count", 중앙값="median",
                                        초과10=lambda s: int((s > 10).sum()),
                                        초과100=lambda s: int((s > 100).sum()))
print(t.to_string(float_format=lambda v: f"{v:,.3f}"))
big = d[d[DR] > 10]
print(f"\n원본에서 소득>1 & DebtRatio>10 인 {len(big):,}행의 소득 분포")
print(big[INC].describe().to_string(float_format=lambda v: f"{v:,.1f}"))
print("해석")
print("  - 소득이 작은 구간(예: 100 이하)에서만 중앙값이 크면 -> 소득 기준을 그 구간까지 올리면 해결")
print("  - 소득이 큰 구간에도 DebtRatio>10 이 고르게 있으면 -> 소득 기준으로는 안 잡힘, 값 기준 규칙 필요")

# ----------------------------------------------------------------------
title("C) 시뮬레이터가 만든 건가: 씬파일러 값 교체로 소득-DebtRatio 짝이 깨진 행")
if not aligned:
    print(f"원본({len(raw):,})과 시뮬레이터({len(sim):,}) 행 수가 달라 행 단위 비교 불가")
else:
    swapped = ~np.isclose(raw[DR].to_numpy(), sim[DR].to_numpy(), equal_nan=True)
    inc_same = np.isclose(raw[INC].fillna(-1).to_numpy(), sim[INC].fillna(-1).to_numpy())
    print(f"DebtRatio가 원본과 달라진 행: {int(swapped.sum()):,} (소득은 그대로인 행: {int((swapped & inc_same).sum()):,})")

    s_big = (sim[INC] > 1) & (sim[DR] > 10)
    r_big = (raw[INC] > 1) & (raw[DR] > 10)
    print(f"\n소득>1 & DebtRatio>10 : 원본 {int(r_big.sum()):,}행 -> 시뮬레이터 {int(s_big.sum()):,}행")
    print(f"  그중 원본에도 원래 있던 행        : {int((s_big & r_big).sum()):,}")
    print(f"  그중 시뮬레이터 교체로 새로 생긴 행: {int((s_big & ~r_big & swapped).sum()):,}")
    print(f"  (원본엔 있었는데 교체로 사라진 행  : {int((r_big & ~s_big).sum()):,})")

    new = s_big & ~r_big & swapped
    if new.any():
        print("\n새로 생긴 행들의 씬파일러 여부:", sim.loc[new, "is_thin_filer"].value_counts().to_dict())
        print("새로 생긴 행들의 DebtRatio 분포:")
        print(sim.loc[new, DR].describe().to_string(float_format=lambda v: f"{v:,.1f}"))
    print("해석")
    print("  - '교체로 새로 생긴 행'이 많고 전부 씬파일러면 -> 시뮬레이터에서 DebtRatio와 소득을 같이 교체하도록 수정")
    print("  - '원래 있던 행'이 대부분이면 -> 원본 데이터 문제, B 결과로 소득 기준 또는 값 기준 결정")