from pathlib import Path


def test_streamlit_app_has_title_and_copy():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    contents = app_path.read_text(encoding="utf-8")

    assert "st.title" in contents
    assert "Let's start building!" in contents
