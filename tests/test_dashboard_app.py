"""tests/test_dashboard_app.py - API 서버가 없어도 대시보드가 죽지 않는지 확인 (HTTP 전용 구조)"""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")          # 아무도 듣지 않는 포트
    return st_testing.AppTest.from_file(str(APP), default_timeout=60)


def test_app_renders_five_tabs_without_api(app):
    at = app.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["고객 심사", "XAI 분석", "공정성", "모델 성능", "통계 검증"]


def test_every_money_input_shows_won_note(app):
    """월 소득 입력 아래에 원화(고정 환율, 만원 미만 버림) 안내가 표시된다"""
    at = app.run()
    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "원화" in captions and "만원" in captions


def test_predict_failure_shows_error_not_crash(app):
    at = app.run()
    next(b for b in at.button if b.key == "run_predict").click()
    at.run()
    assert not at.exception
    assert any("연결" in e.value for e in at.error)
