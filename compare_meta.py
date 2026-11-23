# compare_meta.py
# v1.2(데이터 정제 전)와 v1.3(DebtRatio·RevolvingUtilization 정제 후) 모델 성능 비교용 임시 스크립트
# models/model_meta_v1.2.json, model_meta_v1.3.json 을 읽어 주요 지표를 나란히 출력
# 실행: 프로젝트 최상위 폴더에서 python compare_meta.py  (커밋 불필요)

import json

def load(v):
    with open(f"models/model_meta_v{v}.json", encoding="utf-8") as f:
        return json.load(f)

a, b = load("1.2"), load("1.3")
keys = ["test_auc", "test_ks", "psi", "test_auc_thin_filer", "thin_filer_auc_uplift", "cv_auc_mean"]

print(f"{'지표':24s} {'v1.2':>8s} {'v1.3':>8s}")
for k in keys:
    print(f"{k:24s} {a['final_metrics'][k]:8.4f} {b['final_metrics'][k]:8.4f}")
print(f"{'최종 모델':24s} {a['best_model']['algorithm']:>8s} {b['best_model']['algorithm']:>8s}")