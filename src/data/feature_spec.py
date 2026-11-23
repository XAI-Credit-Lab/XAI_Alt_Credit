"""
src/data/feature_spec.py
XAI 기반 대안신용평가 시스템 - 변수 단위 사전 (한 곳에서 관리)

※ 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

왜 필요한가
  - 단위가 화면·API·설명 문장마다 따로 적혀 있으면 서로 어긋난다.
    (예: DebtRatio에 비율 대신 금액이 들어간 문제, 월 소득을 원화로 입력하는 문제)
  - 대시보드 입력 폼, API 거절 사유(/explain), SHAP Waterfall 이 모두 이 파일 하나를 가져다 쓴다.
  - 입력 허용 범위도 여기서만 정의한다. schema.py 는 field_bounds() 로 가져다 쓴다.

단위 근거
  - GMSC 원본 변수: Kaggle 'Give Me Some Credit' Data Dictionary (미국 데이터 → 금액은 달러)
  - 대안 변수·카드 이용 기간: src/data/simulator.py 생성 로직과 주석

SHAP 기여도 부호 (대시보드·리포트에도 같은 문구로 표기)
  - 모델 출력 = 부도 확률(log-odds). Base Value = 학습 데이터 전체의 평균 예측값.
  - 기여도 + : 부도 위험을 높이는 방향 (거절 쪽)
  - 기여도 - : 부도 위험을 낮추는 방향 (승인 쪽)
  - 거절 사유에는 기여도가 + 인 변수만 쓴다.

사용 예
  from src.data.feature_spec import FEATURE_SPEC, input_label, field_bounds, build_rejection_reasons

  # 대시보드
  s = FEATURE_SPEC["DebtRatio"]
  st.number_input(input_label("DebtRatio"), min_value=s["min"], max_value=s["max"], help=s["help"])

  # schema.py
  DebtRatio: float | None = Field(None, **field_bounds("DebtRatio"))

  # API /explain
  build_rejection_reasons(values, shap_values, medians)
  -> {"rejection_reasons": [{"rank": 1, "feature": "DebtRatio", "feature_name_kr": "부채 비율",
        "current_value": 0.98, "reference_value": 0.366, "shap_contribution": 0.454,
        "explanation": "부채 비율이 98%로, 중앙값(37%)보다 높습니다 (기여도: +0.454)"}, ...],
      "total_rejection_factors": 5}
"""

from __future__ import annotations

import math

# ----------------------------------------------------------------------
# 단위 사전
#   label : 화면에 보일 한글 이름 (단위는 넣지 않음, 단위는 unit 에)
#   unit  : 입력 폼 라벨 괄호 안에 붙일 단위·입력 형식
#   kind  : ratio(비율, %로 표시) / amount(금액) / count(개수·횟수) / score(점수)
#   min, max : 입력 허용 범위. None 이면 제한 없음 (schema.py 는 field_bounds() 사용)
#   step  : 대시보드 입력 증감 단위
#   fmt   : 값 표시 형식 (설명 문장·Waterfall)
#   help  : 대시보드 도움말 (입력 실수 방지 문구)
#   words : (기준보다 클 때, 작을 때) 서술어. 없으면 kind 기본값 사용
# ----------------------------------------------------------------------
FEATURE_SPEC: dict[str, dict] = {
    # ---------------- 기존 금융 변수 ----------------
    "RevolvingUtilizationOfUnsecuredLines": {
        "label": "무담보 한도 사용률", "unit": "비율, 예: 0.5", "kind": "ratio",
        # max 10 = preprocessor.py 의 REVOLVING_MAX 와 같아야 함 (10 초과는 기록 오류로 결측 처리)
        "min": 0.0, "max": 10.0, "step": 0.01, "fmt": "{:.0%}",
        "help": "카드·신용대출 사용액 ÷ 신용 한도. 예: 0.5 = 한도의 50% 사용 (50이 아니라 0.5로 입력). "
                "한도를 넘겨 쓴 경우 1보다 클 수 있음",
    },
    "DebtRatio": {
        "label": "부채 비율", "unit": "비율, 예: 0.3", "kind": "ratio",
        # max 10 = preprocessor.py 의 DEBT_RATIO_MAX 와 같아야 함
        "min": 0.0, "max": 10.0, "step": 0.01, "fmt": "{:.0%}",
        "help": "월 부채상환액·생활비 ÷ 월 소득. 금액이 아니라 비율로 입력 (예: 0.3 = 소득의 30%). "
                "월 소득을 입력하지 않으면 반영되지 않음",
    },
    "MonthlyIncome": {
        # TODO: 원화(만원) 입력·변환 방식 확정 후 unit/fmt/help 변경
        "label": "월 소득", "unit": "달러/월", "kind": "amount",
        "min": 0.0, "max": None, "step": 100.0, "fmt": "{:,.0f}달러",
        "help": "세전 월 소득, 미국 달러 기준 (원화 아님). 학습 데이터 중앙값 약 5,400달러",
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "label": "보유 대출·카드 수", "unit": "개", "kind": "count",
        "min": 0, "max": None, "step": 1, "fmt": "{:,.0f}개",
        "help": "현재 보유 중인 대출과 신용카드 개수의 합",
    },
    "NumberRealEstateLoansOrLines": {
        "label": "부동산 대출 수", "unit": "건", "kind": "count",
        "min": 0, "max": None, "step": 1, "fmt": "{:,.0f}건",
        "help": "주택담보대출 등 부동산 관련 대출 건수",
    },
    "NumberOfDependents": {
        "label": "부양 가족 수", "unit": "명", "kind": "count",
        "min": 0, "max": None, "step": 1, "fmt": "{:,.0f}명",
        "help": "본인을 제외한 부양 가족 수",
    },
    "card_history_months": {
        "label": "카드 이용 기간", "unit": "개월", "kind": "count",
        "min": 0, "max": 360, "step": 1, "fmt": "{:,.0f}개월", "words": ("깁니다", "짧습니다"),
        "help": "첫 신용카드 사용 후 지난 기간. 12개월 미만이면 씬파일러로 분류",
    },
    # ---------------- 연체 (입력 선택) ----------------
    "NumberOfTime30-59DaysPastDueNotWorse": {
        "label": "30-59일 연체 횟수", "unit": "최근 2년, 회", "kind": "count",
        "min": 0, "max": 24, "step": 1, "fmt": "{:,.0f}회",
        "help": "최근 2년 동안 30~59일 연체한 횟수 (그보다 오래 연체한 건은 제외)",
    },
    "NumberOfTime60-89DaysPastDueNotWorse": {
        "label": "60-89일 연체 횟수", "unit": "최근 2년, 회", "kind": "count",
        "min": 0, "max": 12, "step": 1, "fmt": "{:,.0f}회",
        "help": "최근 2년 동안 60~89일 연체한 횟수 (그보다 오래 연체한 건은 제외)",
    },
    "NumberOfTimes90DaysLate": {
        "label": "90일 이상 연체 횟수", "unit": "회", "kind": "count",
        "min": 0, "max": 17, "step": 1, "fmt": "{:,.0f}회",
        "help": "90일 이상 연체한 횟수",
    },
    # ---------------- 대안 변수 (simulator.py) ----------------
    "telecom_payment_rate": {
        "label": "통신비 정상납부율", "unit": "비율, 예: 0.9", "kind": "ratio",
        "min": 0.0, "max": 1.0, "step": 0.01, "fmt": "{:.0%}",
        "help": "통신비를 기한 내 정상 납부한 비율. 예: 0.9 = 90%",
    },
    "utility_payment_rate": {
        "label": "공과금 정상납부율", "unit": "비율, 예: 0.9", "kind": "ratio",
        "min": 0.0, "max": 1.0, "step": 0.01, "fmt": "{:.0%}",
        "help": "공과금(전기·가스·수도)을 기한 내 납부한 비율. 예: 0.9 = 90%",
    },
    "spending_consistency": {
        "label": "소비 일관성", "unit": "점, 0~100", "kind": "score",
        "min": 0.0, "max": 100.0, "step": 1.0, "fmt": "{:,.0f}점",
        # simulator.py: clip(70 + 12 × 잠재값, 0, 100). 잠재값은 '금융 성실도' 공통 요인 + 납부 어려움(꼬리)로 생성
        "help": "월별 소비 패턴이 얼마나 일정한지를 나타내는 점수 (0 = 매우 불규칙, 100 = 매우 일정). "
                "평균 약 70점이며, 납부에 어려움을 겪는 고객일수록 낮게 나타남",
    },
    "regular_payment_count": {
        "label": "정기 결제 건수", "unit": "건", "kind": "count",
        "min": 0, "max": None, "step": 1, "fmt": "{:,.0f}건",
        "help": "현재 이용 중인 정기 결제(구독·할부·자동이체 등) 건수 (평균 약 3.5건)",
    },
    "app_login_frequency": {
        "label": "앱 로그인 빈도", "unit": "회/월", "kind": "count",
        "min": 0, "max": None, "step": 1, "fmt": "{:,.0f}회",
        "help": "최근 30일 동안 금융 앱에 로그인한 횟수 (예: 하루 2번이면 약 60회)",
    },
}

MISSING_TEXT = "미입력"
REFERENCE_NAME = "중앙값"            # 비교 기준 = Train 데이터 중앙값 (재학습 시 artifact 로 저장된 값 사용)
SHAP_SIGN_NOTE = "기여도 +는 부도 위험을 높이는 방향, -는 낮추는 방향입니다"
MIN_REASONS, MAX_REASONS = 3, 5     # 금융 규제 대응: 거절 시 최소 3개, API 는 Top 5

_DEFAULT_WORDS = {
    "count": ("많습니다", "적습니다"),
    "ratio": ("높습니다", "낮습니다"),
    "amount": ("높습니다", "낮습니다"),
    "score": ("높습니다", "낮습니다"),
}


# ----------------------------------------------------------------------
# 조사 처리: 받침 유무로 이/가, 은/는, 을/를, (으)로 선택
# ----------------------------------------------------------------------
def _last_jong(word: str) -> int | None:
    """마지막 글자의 받침 번호 (0 = 받침 없음). 판단할 수 없으면 None"""
    ch = word.rstrip()[-1:] if word else ""
    if "가" <= ch <= "힣":
        return (ord(ch) - ord("가")) % 28
    if ch == "%":                     # '퍼센트' -> 받침 없음
        return 0
    if ch.isdigit():                  # 끝자리 읽기 (10=십, 100=백 등도 받침 있음 → 결과 동일)
        return {"0": 21, "1": 8, "2": 0, "3": 16, "4": 0,
                "5": 0, "6": 1, "7": 8, "8": 8, "9": 0}[ch]
    return None


def josa(word: str, pair: str) -> str:
    """word 뒤에 맞는 조사를 붙여 반환. pair: '이/가', '은/는', '을/를', '으로/로'"""
    with_jong, without_jong = pair.split("/")
    jong = _last_jong(word)
    if jong is None:
        return f"{word}{with_jong}({without_jong})"
    if pair == "으로/로":                     # ㄹ 받침(8)은 '로'
        return word + ("로" if jong in (0, 8) else "으로")
    return word + (with_jong if jong else without_jong)


# ----------------------------------------------------------------------
# 화면·문장용 도우미
# ----------------------------------------------------------------------
def is_missing(value) -> bool:
    """None, float NaN, numpy NaN(float32 포함), pandas NA/NaT 를 모두 결측으로 본다"""
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def label(name: str) -> str:
    """한글 이름 (사전에 없으면 원래 이름). API 의 feature_name_kr 로도 사용"""
    return FEATURE_SPEC.get(name, {}).get("label", name)


def input_label(name: str) -> str:
    """대시보드 입력 폼 라벨: '부채 비율 (비율, 예: 0.3)'"""
    s = FEATURE_SPEC.get(name)
    return f"{s['label']} ({s['unit']})" if s else name


def field_bounds(name: str) -> dict:
    """schema.py 의 Field(**field_bounds(name)) 용. 범위를 이 파일 한 곳에서만 관리"""
    s = FEATURE_SPEC[name]
    return {k: v for k, v in (("ge", s["min"]), ("le", s["max"])) if v is not None}


def format_value(name: str, value) -> str:
    """값을 단위와 함께 표시: 0.98 -> '98%', 211 -> '211개월', None/NaN -> '미입력'"""
    if is_missing(value):
        return MISSING_TEXT
    s = FEATURE_SPEC.get(name)
    if s:
        return s["fmt"].format(value)
    return f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)


def _words(name: str) -> tuple[str, str]:
    s = FEATURE_SPEC.get(name, {})
    return s.get("words") or _DEFAULT_WORDS.get(s.get("kind"), ("높습니다", "낮습니다"))


def reason_sentence(name: str, value, median: float | None, shap: float) -> str:
    """거절 사유 자연어 문장 (비교 기준 = Train 중앙값, 같은지 여부는 '화면 표시값' 기준)

    예) '부채 비율이 98%로, 중앙값(37%)보다 높습니다 (기여도: +0.454)'
        '90일 이상 연체 횟수가 2회로, 대부분의 고객(0회)보다 많습니다 (기여도: +0.300)'
        '월 소득이 미입력이어서 중앙값(5,400달러)으로 계산했습니다 (기여도: +0.251)'
    """
    lab, v_txt = label(name), format_value(name, value)
    tail = f" (기여도: {shap:+.3f})"
    if v_txt == MISSING_TEXT:
        base = (f"{REFERENCE_NAME}({format_value(name, median)})으로 계산했습니다"
                if not is_missing(median) else "계산에서 제외했습니다")
        return f"{josa(lab, '이/가')} 미입력이어서 {base}{tail}"
    if is_missing(median):
        return f"{josa(lab, '이/가')} {v_txt}입니다{tail}"

    m_txt = format_value(name, median)
    # 중앙값이 0인 횟수 변수(연체 등)는 '대부분의 고객(0회)'으로 표현
    ref = "대부분의 고객" if median == 0 and FEATURE_SPEC.get(name, {}).get("kind") == "count" else REFERENCE_NAME
    if v_txt == m_txt:                               # 표시 자릿수에서 같으면 '같습니다'
        return f"{josa(lab, '이/가')} {josa(v_txt, '으로/로')}, {ref}({m_txt})과 같습니다{tail}"
    more, less = _words(name)
    return f"{josa(lab, '이/가')} {josa(v_txt, '으로/로')}, {ref}({m_txt})보다 {more if value > median else less}{tail}"


def build_rejection_reasons(values: dict, shap_values: dict, medians: dict,
                            top_k: int = MAX_REASONS) -> dict:
    """/explain 응답용 거절 사유 Top K (과제 명세 결과 예시 형식)

    - 기여도가 + (부도 위험을 높이는 방향)인 변수만 사유로 사용, 기여도 큰 순서
    - 결측 지시변수 등 사전에 없는 변수는 제외
    - total_rejection_factors: 기여도 + 인 전체 변수 수
    """
    adverse = sorted(
        ((f, s) for f, s in shap_values.items() if f in FEATURE_SPEC and s > 0),
        key=lambda x: x[1], reverse=True,
    )
    reasons = []
    for rank, (f, s) in enumerate(adverse[:top_k], start=1):
        v, m = values.get(f), medians.get(f)
        reasons.append({
            "rank": rank,
            "feature": f,
            "feature_name_kr": label(f),
            "current_value": None if is_missing(v) else v,
            "reference_value": None if is_missing(m) else m,
            "shap_contribution": round(float(s), 4),
            "explanation": reason_sentence(f, v, m, float(s)),
        })
    return {"rejection_reasons": reasons, "total_rejection_factors": len(adverse)}


if __name__ == "__main__":
    # 동작 확인: python -m src.data.feature_spec
    print("[입력 폼 라벨]")
    for n in FEATURE_SPEC:
        print(f"  {input_label(n)}")
    print("\n[거절 사유 문장 예시]")
    print(" ", reason_sentence("DebtRatio", 0.9844, 0.366, 0.454))
    print(" ", reason_sentence("RevolvingUtilizationOfUnsecuredLines", 0.5144, 0.151, 0.483))
    print(" ", reason_sentence("MonthlyIncome", 2308.0, 5400.0, 0.251))
    print(" ", reason_sentence("card_history_months", 6, 193, 0.111))
    print(" ", reason_sentence("NumberOfOpenCreditLinesAndLoans", 8, 8, 0.02))
    print(" ", reason_sentence("NumberOfTimes90DaysLate", 2, 0, 0.3))
    print(" ", reason_sentence("app_login_frequency", 3, 49, 0.12))
    print(" ", reason_sentence("DebtRatio", None, 0.366, 0.01))
    print(" ", reason_sentence("MonthlyIncome", 2308.0, None, 0.2))
    print(" ", reason_sentence("spending_consistency", 14.0, 53.07, 0.5))
    print("\n[/explain 응답 예시]")
    import json
    print(json.dumps(build_rejection_reasons(
        {"DebtRatio": 0.98, "telecom_payment_rate": 0.78, "NumberOfTimes90DaysLate": 3, "MonthlyIncome": 6000},
        {"DebtRatio": 0.454, "telecom_payment_rate": 0.15, "NumberOfTimes90DaysLate": 0.12,
         "MonthlyIncome": -0.08, "DebtRatio_missing": 0.05},
        {"DebtRatio": 0.366, "telecom_payment_rate": 0.92, "NumberOfTimes90DaysLate": 0, "MonthlyIncome": 5400},
    ), ensure_ascii=False, indent=2))