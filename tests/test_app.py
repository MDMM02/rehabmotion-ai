from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_with_medical_disclaimer() -> None:
    app = AppTest.from_file("app/main.py").run()

    assert not app.exception
    assert app.title[0].value == "RehabMotion AI"
    assert "not a certified medical device" in app.warning[0].value

