"""End-to-end tests that drive the real Streamlit app (headless) with streamlit.testing."""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture()
def app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_starts_without_exception(app):
    assert not app.exception
    assert len(app.tabs) == 5


def test_five_kpis_present(app):
    m = metrics(app)
    for label in ["Total Sales", "Total Profit", "Total Orders", "Total Quantity Sold", "Profit Margin"]:
        assert label in m, label


def test_at_least_three_filters(app):
    assert len(app.sidebar.multiselect) >= 3 and len(app.sidebar.date_input) >= 1


def test_filter_updates_kpis(app):
    before = metrics(app)["Total Sales"]
    region = next(ms for ms in app.sidebar.multiselect if ms.label == "Region")
    region.set_value([region.options[0]]).run()
    assert not app.exception
    assert metrics(app)["Total Sales"] != before


def test_empty_filter_shows_warning_not_crash(app):
    region = next(ms for ms in app.sidebar.multiselect if ms.label == "Region")
    region.set_value([]).run()
    assert not app.exception
    assert any("No records match" in w.value for w in app.warning)


def test_cleaning_option_changes_data(app):
    before = metrics(app)["Total Orders"]
    app.sidebar.radio[0].set_value("Impute with Sub-Category median").run()
    assert not app.exception
    assert metrics(app)["Total Orders"] != before


def test_welcome_screen_when_no_data():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.toggle[0].set_value(False).run()
    assert not at.exception
    assert any("Upload" in i.value for i in at.info)
