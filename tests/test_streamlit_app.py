import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from streamlit_app import (  # noqa: E402
    build_checks,
    build_csv,
    check_github,
    check_gitlab,
    check_wikimedia,
    run_checks_for_username,
)


def _json_response(payload, status_code=200):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


@patch("streamlit_app.requests.get")
def test_check_github_found(mock_get):
    mock_get.return_value = _json_response(
        {"html_url": "https://github.com/octocat", "public_repos": 8, "followers": 99}
    )

    result = check_github("octocat")

    assert result.found is True
    assert result.source == "GitHub"
    assert result.status == "found"


@patch("streamlit_app.requests.get")
def test_check_gitlab_not_found(mock_get):
    mock_get.return_value = _json_response([])

    result = check_gitlab("missing")

    assert result.found is False
    assert result.status == "not found"


@patch("streamlit_app.requests.get")
def test_check_wikimedia_missing_user(mock_get):
    mock_get.return_value = _json_response({"query": {"users": [{"name": "x", "missing": ""}]}})

    result = check_wikimedia("x")

    assert result.found is False
    assert result.status == "not found"


@patch("streamlit_app.build_checks")
def test_run_checks_for_username_parallel(mock_build_checks):
    checker_a = Mock(return_value=Mock(source="A"))
    checker_b = Mock(return_value=Mock(source="B"))
    mock_build_checks.return_value = [checker_a, checker_b]

    results = run_checks_for_username("abc", workers=2)

    assert len(results) == 2
    assert {item.source for item in results} == {"A", "B"}
    checker_a.assert_called_once_with("abc")
    checker_b.assert_called_once_with("abc")


def test_build_csv_has_headers_and_rows():
    csv_text = build_csv(
        [
            {
                "username": "octocat",
                "source": "GitHub",
                "found": "yes",
                "status": "found",
                "summary": "Repos: 8",
                "profile_url": "https://github.com/octocat",
            }
        ]
    )

    assert "username,source,found,status,summary,profile_url" in csv_text
    assert "octocat,GitHub,yes,found" in csv_text


def test_build_checks_includes_core_sources():
    sources = {check.__name__ for check in build_checks()}

    assert "check_github" in sources
    assert "check_gitlab" in sources
    assert "check_reddit" in sources
    assert "check_hackernews" in sources
