from __future__ import annotations

import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import requests
import streamlit as st

REQUEST_TIMEOUT = 10
DEFAULT_UA = "public-presence-checker/2.0"


@dataclass
class SourceResult:
    source: str
    found: bool
    status: str
    profile_url: str = ""
    summary: str = ""


def fetch_json(url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[Optional[Dict], Optional[str], int]:
    merged_headers = {"User-Agent": DEFAULT_UA, "Accept": "application/json", **(headers or {})}
    try:
        response = requests.get(url, headers=merged_headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return None, str(exc), 0

    status_code = response.status_code
    if status_code == 404:
        return None, None, 404
    if status_code >= 400:
        return None, f"http {status_code}", status_code

    try:
        return response.json(), None, status_code
    except ValueError:
        return None, "invalid json", status_code


def check_github(username: str) -> SourceResult:
    data, error, code = fetch_json(f"https://api.github.com/users/{username}")
    if code == 404:
        return SourceResult("GitHub", False, "not found")
    if error or data is None:
        return SourceResult("GitHub", False, f"error: {error or 'unknown'}")

    summary = f"Repos: {data.get('public_repos', 0)} | Followers: {data.get('followers', 0)}"
    return SourceResult("GitHub", True, "found", data.get("html_url", ""), summary)


def check_gitlab(username: str) -> SourceResult:
    data, error, code = fetch_json(f"https://gitlab.com/api/v4/users?username={username}")
    if error or data is None:
        return SourceResult("GitLab", False, f"error: {error or f'http {code}'}")

    if not data:
        return SourceResult("GitLab", False, "not found")

    user = data[0]
    summary = f"State: {user.get('state', 'unknown')}"
    return SourceResult("GitLab", True, "found", user.get("web_url", ""), summary)


def check_reddit(username: str) -> SourceResult:
    data, error, code = fetch_json(f"https://www.reddit.com/user/{username}/about.json")
    if code == 404:
        return SourceResult("Reddit", False, "not found")
    if error or data is None:
        return SourceResult("Reddit", False, f"error: {error or 'unknown'}")

    user_data = data.get("data", {})
    profile = f"https://www.reddit.com/user/{username}/"
    summary = f"Karma: {user_data.get('total_karma', 0)}"
    return SourceResult("Reddit", True, "found", profile, summary)


def check_hackernews(username: str) -> SourceResult:
    data, error, code = fetch_json(f"https://hacker-news.firebaseio.com/v0/user/{username}.json")
    if code == 404 or data is None:
        return SourceResult("Hacker News", False, "not found" if not error else f"error: {error}")

    summary = f"Karma: {data.get('karma', 0)}"
    profile = f"https://news.ycombinator.com/user?id={username}"
    return SourceResult("Hacker News", True, "found", profile, summary)


def check_keybase(username: str) -> SourceResult:
    data, error, code = fetch_json(f"https://keybase.io/_/api/1.0/user/lookup.json?usernames={username}")
    if error or data is None:
        return SourceResult("Keybase", False, f"error: {error or f'http {code}'}")

    them = data.get("them", [])
    if not them:
        return SourceResult("Keybase", False, "not found")

    profile = f"https://keybase.io/{username}"
    summary = "Public identity available"
    return SourceResult("Keybase", True, "found", profile, summary)


def check_wikimedia(username: str) -> SourceResult:
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=users&ususers={username}&format=json"
    )
    data, error, code = fetch_json(url)
    if error or data is None:
        return SourceResult("Wikimedia", False, f"error: {error or f'http {code}'}")

    users = data.get("query", {}).get("users", [])
    if not users or "missing" in users[0]:
        return SourceResult("Wikimedia", False, "not found")

    profile = f"https://en.wikipedia.org/wiki/User:{username}"
    summary = "Wikipedia account exists"
    return SourceResult("Wikimedia", True, "found", profile, summary)


def check_devto(username: str) -> SourceResult:
    data, error, code = fetch_json(f"https://dev.to/api/users/by_username?url={username}")
    if code == 404:
        return SourceResult("DEV Community", False, "not found")
    if error or data is None:
        return SourceResult("DEV Community", False, f"error: {error or f'http {code}'}")

    summary = f"Articles: {data.get('articles_count', 0)}"
    return SourceResult("DEV Community", True, "found", f"https://dev.to/{username}", summary)


def check_pypi_search(query: str) -> SourceResult:
    url = f"https://pypi.org/search/?q={query}"
    try:
        response = requests.get(url, headers={"User-Agent": DEFAULT_UA}, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return SourceResult("PyPI search", False, f"error: {exc}")

    if response.status_code >= 400:
        return SourceResult("PyPI search", False, f"error: http {response.status_code}")

    found = "package-snippet" in response.text
    summary = "Potential package matches" if found else "No package matches"
    return SourceResult("PyPI search", found, "found" if found else "not found", url, summary)


def build_checks() -> List[Callable[[str], SourceResult]]:
    return [
        check_github,
        check_gitlab,
        check_reddit,
        check_hackernews,
        check_keybase,
        check_wikimedia,
        check_devto,
        check_pypi_search,
    ]


def run_checks_for_username(username: str, workers: int = 6) -> List[SourceResult]:
    checkers = build_checks()
    results: List[SourceResult] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(checkers))) as executor:
        futures = [executor.submit(checker, username) for checker in checkers]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item.source.lower())


def build_csv(rows: List[Dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["username", "source", "found", "status", "summary", "profile_url"])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


st.set_page_config(page_title="Public Presence Checker Pro", page_icon="🔎", layout="wide")
st.title("🔎 Public Presence Checker Pro")
st.caption("Fast, consent-first checks across lawful public sources. No private API keys required.")

with st.expander("Legal + ethical use", expanded=False):
    st.write(
        "Use only with permission, legitimate security/self-audit workflows, or your own accounts. "
        "Do not use for stalking, harassment, doxxing, or non-consensual surveillance."
    )

left, right = st.columns([2, 1])
with left:
    usernames_raw = st.text_area(
        "Usernames (comma/newline separated)",
        value="octocat",
        height=120,
        placeholder="octocat, torvalds, someone_else",
    )
with right:
    max_workers = st.slider("Parallel workers", min_value=1, max_value=12, value=6)
    run = st.button("Run checks", type="primary", width="stretch")

if run:
    usernames = [item.strip() for item in usernames_raw.replace("\n", ",").split(",") if item.strip()]
    usernames = list(dict.fromkeys(usernames))

    if not usernames:
        st.error("Enter at least one username.")
    else:
        all_rows: List[Dict[str, str]] = []
        progress = st.progress(0)

        for idx, username in enumerate(usernames, start=1):
            with st.spinner(f"Checking {username}..."):
                results = run_checks_for_username(username, workers=max_workers)

            found_count = sum(1 for item in results if item.found)
            st.subheader(f"@{username} — {found_count}/{len(results)} sources matched")

            table_rows = []
            for result in results:
                table_rows.append(
                    {
                        "source": result.source,
                        "found": "yes" if result.found else "no",
                        "status": result.status,
                        "summary": result.summary,
                        "profile_url": result.profile_url,
                    }
                )
                all_rows.append({"username": username, **table_rows[-1]})

            st.dataframe(table_rows, width="stretch")
            progress.progress(int((idx / len(usernames)) * 100))

        csv_data = build_csv(all_rows)
        json_data = json.dumps(all_rows, indent=2)

        st.divider()
        st.subheader("Export results")
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name="public_presence_results.csv",
                mime="text/csv",
                width="stretch",
            )
        with col_b:
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name="public_presence_results.json",
                mime="application/json",
                width="stretch",
            )

        total_sources = len(build_checks()) * len(usernames)
        total_found = sum(1 for row in all_rows if row["found"] == "yes")
        st.success(f"Completed: {total_found}/{total_sources} source checks matched.")
