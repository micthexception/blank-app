import sys
from pathlib import Path
from unittest.mock import Mock, patch

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from streamlit_app import check_github, check_hackernews, check_reddit, run_checks


def _mock_response(payload, status_code=200):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status.side_effect = None if status_code < 400 else requests.HTTPError(response=resp)
    return resp


@patch("streamlit_app.requests.get")
def test_check_github_found(mock_get):
    mock_get.return_value = _mock_response(
        {"html_url": "https://github.com/octocat", "public_repos": 8, "followers": 99}
    )

    result = check_github("octocat")

    assert result.found is True
    assert result.source == "GitHub"
    assert "Public repos" in result.summary


@patch("streamlit_app.requests.get")
def test_check_reddit_not_found(mock_get):
    response = _mock_response({}, status_code=404)
    mock_get.return_value = response

    result = check_reddit("missing_user")

    assert result.found is False
    assert result.status == "not found"


@patch("streamlit_app.requests.get")
def test_check_hackernews_none(mock_get):
    mock_get.return_value = _mock_response(None)

    result = check_hackernews("unknown")

    assert result.found is False
    assert result.status == "not found"


@patch("streamlit_app.check_pypi")
@patch("streamlit_app.check_hackernews")
@patch("streamlit_app.check_reddit")
@patch("streamlit_app.check_github")
def test_run_checks_runs_all(mock_gh, mock_reddit, mock_hn, mock_pypi):
    mock_gh.return_value.source = "GitHub"
    mock_reddit.return_value.source = "Reddit"
    mock_hn.return_value.source = "Hacker News"
    mock_pypi.return_value.source = "PyPI search"

    results = run_checks("abc")

    assert len(results) == 4
    assert {item.source for item in results} == {"GitHub", "Reddit", "Hacker News", "PyPI search"}
