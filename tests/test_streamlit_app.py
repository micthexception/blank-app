import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from streamlit_app import (  # noqa: E402
    build_checks,
    build_csv,
    check_dockerhub,
    check_github,
    check_npm_maintainer,
    check_stackoverflow,
    run_checks_for_username,
    timeline_summary,
)


def _json_response(payload, status_code=200):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


@patch("streamlit_app.requests.get")
def test_check_github_found(mock_get):
    mock_get.return_value = _json_response(
        {
            "html_url": "https://github.com/octocat",
            "public_repos": 8,
            "followers": 99,
            "created_at": "2011-01-25T18:44:36Z",
        }
    )
    result = check_github("octocat")
    assert result.found is True
    assert result.source == "GitHub"
    assert result.status == "found"


@patch("streamlit_app.requests.get")
def test_check_stackoverflow_not_found(mock_get):
    mock_get.return_value = _json_response({"items": []})
    result = check_stackoverflow("missing")
    assert result.found is False
    assert result.status == "not found"


@patch("streamlit_app.requests.get")
def test_check_dockerhub_not_found(mock_get):
    mock_get.return_value = _json_response({}, status_code=404)
    result = check_dockerhub("missing")
    assert result.found is False
    assert result.status == "not found"


@patch("streamlit_app.requests.get")
def test_check_npm_maintainer_found(mock_get):
    mock_get.return_value = _json_response({"total": 2})
    result = check_npm_maintainer("alice")
    assert result.found is True
    assert "Matches: 2" in result.summary


@patch("streamlit_app.build_checks")
def test_run_checks_for_username_parallel(mock_build_checks):
    checker_a = Mock(return_value=Mock(source="A"))
    checker_b = Mock(return_value=Mock(source="B"))
    mock_build_checks.return_value = [checker_a, checker_b]

    results = run_checks_for_username("abc", workers=2)

    assert len(results) == 2
    assert {item.source for item in results} == {"A", "B"}


def test_build_csv_has_headers_and_rows():
    csv_text = build_csv(
        [
            {
                "username": "octocat",
                "source": "GitHub",
                "found": "yes",
                "status": "found",
                "summary": "Repos: 8",
                "created_at": "2011-01-25T18:44:36+00:00",
                "profile_url": "https://github.com/octocat",
            }
        ]
    )
    assert "username,source,found,status,summary,created_at,profile_url" in csv_text
    assert "octocat,GitHub,yes,found" in csv_text


def test_timeline_summary_with_dates():
    from streamlit_app import SourceResult

    results = [
        SourceResult("A", True, "found", created_at="2020-01-01T00:00:00+00:00"),
        SourceResult("B", True, "found", created_at="2022-01-01T00:00:00+00:00"),
    ]
    summary = timeline_summary(results)
    assert "Earliest visible account creation" in summary


def test_build_checks_count_grew():
    assert len(build_checks()) >= 12
