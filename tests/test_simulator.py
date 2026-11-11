from pathlib import Path
import sys # 파이썬이 모듈을 찾는 경로 목록(sys.path)에 프로젝트 루트 경로를 추가하기 위해 사용

import pytest # 테스트 도구

PROJECT_ROOT = Path(__file__).resolve().parents[1] # resolve() : 절대경로로 변경
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# simulator.py에서 필요한 것만 가져오기
from src.data.simulator import (
    ALT_COLS,
    CONTROL_BIAS_RATIO,
    SCENARIO_BIAS_RATIO,
    add_alt_features,
    approval_report,
    feature_di_report,
    load_base,
)

# 원본 GMSC 데이터(나이 0 제거)를 준비하는 fixture
# scope="module": 이 파일 전체에서 한 번만 읽고 모든 테스트가 재사용 (매번 15만 행을 읽지 않도록)
@pytest.fixture(scope="module")
def base():
    return load_base()

# 통제용(bias 0)·시나리오용(bias 1.0) 데이터를 한 번만 만들어 여러 테스트가 공유
# main()이 저장하는 결과_통제.csv / 결과_시나리오.csv와 같은 데이터
@pytest.fixture(scope="module")
def generated_datasets(base):
    return {
        "control": add_alt_features(base, bias_ratio=CONTROL_BIAS_RATIO),
        "scenario": add_alt_features(base, bias_ratio=SCENARIO_BIAS_RATIO),
    }

# 대안변수와 부도의 |상관|이 0.3~0.5인지 (통제용·시나리오용 각각 1번씩, 총 2번 실행)
@pytest.mark.parametrize("scenario", ["control", "scenario"])
def test_alt_feature_correlations_stay_within_spec(generated_datasets, scenario):
    result = generated_datasets[scenario]
    correlations = result[ALT_COLS].corrwith(result["SeriousDlqin2yrs"]).abs()

    # 실패하면 변수별 상관값이 출력
    assert correlations.between(0.3, 0.5).all(), correlations.to_dict()

# bias_ratio를 올리면 성별 DI 감소하는지 (대안변수 기준·승인 라벨 기준 둘 다)
def test_gender_disparate_impact_decreases_as_bias_increases(base, capsys):
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = [add_alt_features(base, bias_ratio=br) for br in ratios]

    feature_dis = [feature_di_report(r, verbose=False) for r in results]
    approval_dis = [approval_report(r) for r in results]
    capsys.readouterr()  # approval_report 출력 숨김

    # bias_ratio를 올릴 때마다 DI가 계속 내려가야 함
    assert all(a > b for a, b in zip(feature_dis, feature_dis[1:])), feature_dis
    assert all(a > b for a, b in zip(approval_dis, approval_dis[1:])), approval_dis

# 시나리오(bias_ratio=1.0)에서 성별 DI < 0.8
def test_scenario_gender_di_below_threshold(generated_datasets):
    di = feature_di_report(generated_datasets["scenario"], verbose=False)
    assert di < 0.8, f"scenario feature DI = {di:.3f}"

# thin_filer_ratio 반영
@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.3])
def test_thin_filer_ratio_matches_requested_ratio(base, ratio):
    result = add_alt_features(base, thin_filer_ratio=ratio)
    assert result["is_thin_filer"].mean() == pytest.approx(ratio, abs=0.005)

# 직접 실행 (종료 코드 0: 전부 통과, 1: 실패한 테스트 있음)
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))