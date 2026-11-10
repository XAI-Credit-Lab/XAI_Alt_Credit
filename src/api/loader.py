# json파일, auc, ks 등 로드하는 코드

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2] # BASE_DIR: 이 파일이 들어있는 폴더의 상위상위 폴더

# /health의 "model_version" -> "model_meta_v1.0.json" 파일에서 로드
def load_health_model_version():
    with open(BASE_DIR/"models"/"model_meta_v1.0.json", encoding = "utf-8") as f:
            model_json = json.load(f)
            return model_json["best_model"]["algorithm"]


