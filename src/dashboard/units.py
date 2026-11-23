"""
src/dashboard/units.py
대시보드 단위·금액 표기 도우미 (화면 전용, 모델·API와 무관)

규칙
  - 금액은 달러가 기본. 바로 아래에 원화를 고정 환율로 바꿔 함께 표기한다.
  - 원화는 만원 미만을 버린다. (예: 3,000달러 -> 4,040,400원 -> '404만원')
  - 모든 숫자에는 단위를 붙인다. (%, %p, 점, 개월, 배, 명 ...)

고정 환율
  - 2026년 11월 23일 기준 1달러 = 1,346.80원 (고정값). 예: 3,000달러 -> 4,040,400원 -> '404만원'
  - 환율을 바꾸려면 USD_KRW 와 FX_DATE 만 고치면 된다.
  - 만원 미만을 버리지 않고 '404만 400원'처럼 원 단위까지 보이려면 KRW_FLOOR_UNIT = 1 로 바꾼다.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR

FX_DATE = "2026년 11월 23일"        # 환율 기준일
USD_KRW = Decimal("1346.80")      # 1달러 = 1,346.80원 (고정값)
KRW_FLOOR_UNIT = 10_000           # 이 단위 미만은 버림 (10,000 = 만원 미만 버림)

FX_NOTE = f"{FX_DATE} 기준 환율 1달러 = {USD_KRW:.2f}원 고정 · 원화는 만원 미만 버림"


def usd(value, decimals: int = 0) -> str:
    """달러 표기: 3000 -> '$3,000'"""
    return f"${float(value):,.{decimals}f}"


def usd_to_krw(value) -> int:
    """달러 -> 원화(고정 환율). KRW_FLOOR_UNIT 미만은 버린다."""
    won = Decimal(str(value)) * USD_KRW
    unit = Decimal(KRW_FLOOR_UNIT)
    return int((won / unit).to_integral_value(rounding=ROUND_FLOOR) * unit)


def krw_text(won: int) -> str:
    """원화 표기: 4,040,000 -> '404만원', 123,450,000 -> '1억 2,345만원', 0 -> '1만원 미만'"""
    won = int(won)
    if won <= 0:
        return "1만원 미만" if KRW_FLOOR_UNIT >= 10_000 else "0원"
    eok, rest = divmod(won, 100_000_000)
    man, won_rest = divmod(rest, 10_000)
    parts = []
    if eok:
        parts.append(f"{eok:,}억")
    if man:
        parts.append(f"{man:,}만")
    if won_rest:                                  # KRW_FLOOR_UNIT < 10,000 일 때만 남음
        parts.append(f"{won_rest}")
    text = " ".join(parts)
    return text + ("원" if not text.endswith("원") else "")


def krw(value_usd) -> str:
    """달러 금액을 원화 문자열로: 3000 -> '404만원'"""
    return krw_text(usd_to_krw(value_usd))


def money_lines(value_usd, decimals: int = 0) -> tuple[str, str]:
    """(달러 표기, 원화 표기): 3000 -> ('$3,000', '원화 : 404만원')"""
    return usd(value_usd, decimals), f"원화 : {krw(value_usd)}"


def pct(x, digits: int = 1) -> str:
    """비율(0~1) -> '12.3%'"""
    return f"{float(x) * 100:.{digits}f}%"


def pp(x, digits: int = 1, signed: bool = False) -> str:
    """비율 차이(0~1) -> '+1.2%p' (퍼센트포인트)"""
    return f"{float(x) * 100:{'+' if signed else ''}.{digits}f}%p"


def years_months(months: int) -> str:
    """72 -> '6년 0개월', 8 -> '8개월'"""
    months = int(months)
    y, m = divmod(months, 12)
    return f"{y}년 {m}개월" if y else f"{m}개월"
