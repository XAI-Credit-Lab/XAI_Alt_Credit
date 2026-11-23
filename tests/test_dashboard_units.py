"""tests/test_dashboard_units.py - 대시보드 금액·단위 표기 테스트 (API·모델 없이 실행)"""
import pytest

from src.dashboard import units


def test_fixed_rate_and_note_text():
    """2026년 11월 23일 기준 1달러 = 1,346.80원 고정"""
    assert float(units.USD_KRW) == 1346.80
    assert units.FX_NOTE == "2026년 11월 23일 기준 환율 1달러 = 1346.80원 고정 · 원화는 만원 미만 버림"


def test_krw_floors_below_man_won():
    """원화는 만원 미만 버림: 3,000달러 -> 4,040,400원 -> 404만원"""
    assert units.usd_to_krw(3000) == 4_040_000
    assert units.krw(3000) == "404만원"
    assert units.money_lines(3000) == ("$3,000", "원화 : 404만원")


@pytest.mark.parametrize("dollars, expected", [
    (5, "1만원 미만"),          # 6,734원 -> 만원 미만은 버려서 0
    (5000, "673만원"),
    (100000, "1억 3,468만원"),
    (0, "1만원 미만"),
])
def test_krw_text_formats(dollars, expected):
    assert units.krw(dollars) == expected


def test_floor_unit_can_show_won(monkeypatch):
    """KRW_FLOOR_UNIT=1 이면 원 단위까지 표기 (예: 404만 400원)"""
    monkeypatch.setattr(units, "KRW_FLOOR_UNIT", 1)
    assert units.krw(3000) == "404만 400원"


def test_pct_pp_and_period_units():
    assert units.pct(0.0923) == "9.2%"
    assert units.pp(0.186) == "18.6%p"
    assert units.pp(-0.104, signed=True) == "-10.4%p"
    assert units.years_months(60) == "5년 0개월"
    assert units.years_months(8) == "8개월"


def test_gauge_svg_has_no_blank_lines():
    """빈 줄이 있으면 st.markdown 이 SVG 를 글자로 깨뜨린다 (결과가 없을 때 포인터가 비는 경우 포함)"""
    from src.dashboard.theme import gauge_svg
    for score in (None, 0, 908, 1000):
        svg = gauge_svg(score, 1000, "A", "A등급")
        assert svg.lstrip().startswith("<svg") and svg.rstrip().endswith("</svg>")
        assert all(line.strip() for line in svg.splitlines())
