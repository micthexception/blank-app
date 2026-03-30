from __future__ import annotations

import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import requests
import streamlit as st

REQUEST_TIMEOUT = 10
DEFAULT_UA = "public-presence-checker/3.0"


@dataclass
class SourceResult:
    source: str
    found: bool
    status: str
    profile_url: str = ""
    summary: str = ""
    created_at: str = ""


def iso_to_age(iso_value: str) -> str:
    if not iso_value:
        return ""
    try:
        parsed = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
        years = max(delta.days // 365, 0)
        return f"{years}y" if years else "<1y"
    except ValueError:
        return ""


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
    created = data.get("created_at", "")
    summary = f"Repos: {data.get('public_repos', 0)} | Followers: {data.get('followers', 0)} | Age: {iso_to_age(created)}"
    return SourceResult("GitHub", True, "found", data.get("html_url", ""), summary, created)


def check_gitlab(username: str) -> SourceResult:
    data, error, _ = fetch_json(f"https://gitlab.com/api/v4/users?username={username}")
    if error or data is None:
        return SourceResult("GitLab", False, f"error: {error or 'unknown'}")
    if not data:
        return SourceResult("GitLab", False, "not found")
    user = data[0]
    created = user.get("created_at", "")
    summary = f"State: {user.get('state', 'unknown')} | Age: {iso_to_age(created)}"
    return SourceResult("GitLab", True, "found", user.get("web_url", ""), summary, created)


def check_reddit(username: str) -> SourceResult:
    data, error, code = fetch_json(f"https://www.reddit.com/user/{username}/about.json")
    if code == 404:
        return SourceResult("Reddit", False, "not found")
    if error or data is None:
        return SourceResult("Reddit", False, f"error: {error or 'unknown'}")
    user_data = data.get("data", {})
    created_utc = user_data.get("created_utc")
    created = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat() if created_utc else ""
    summary = f"Karma: {user_data.get('total_karma', 0)} | Age: {iso_to_age(created)}"
    return SourceResult("Reddit", True, "found", f"https://www.reddit.com/user/{username}/", summary, created)


def check_hackernews(username: str) -> SourceResult:
    data, error, code = fetch_json(f"https://hacker-news.firebaseio.com/v0/user/{username}.json")
    if code == 404 or data is None:
        return SourceResult("Hacker News", False, "not found" if not error else f"error: {error}")
    created_unix = data.get("created")
    created = datetime.fromtimestamp(created_unix, tz=timezone.utc).isoformat() if created_unix else ""
    summary = f"Karma: {data.get('karma', 0)} | Age: {iso_to_age(created)}"
    return SourceResult("Hacker News", True, "found", f"https://news.ycombinator.com/user?id={username}", summary, created)


def check_keybase(username: str) -> SourceResult:
    data, error, _ = fetch_json(f"https://keybase.io/_/api/1.0/user/lookup.json?usernames={username}")
    if error or data is None:
        return SourceResult("Keybase", False, f"error: {error or 'unknown'}")
    if not data.get("them"):
        return SourceResult("Keybase", False, "not found")
    return SourceResult("Keybase", True, "found", f"https://keybase.io/{username}", "Public identity available")


def check_wikimedia(username: str) -> SourceResult:
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=users&ususers={username}&format=json"
    data, error, _ = fetch_json(url)
    if error or data is None:
        return SourceResult("Wikimedia", False, f"error: {error or 'unknown'}")
    users = data.get("query", {}).get("users", [])
    if not users or "missing" in users[0]:
        return SourceResult("Wikimedia", False, "not found")
    return SourceResult("Wikimedia", True, "found", f"https://en.wikipedia.org/wiki/User:{username}", "Wikipedia account exists")


def check_devto(username: str) -> SourceResult:
    data, error, code = fetch_json(f"https://dev.to/api/users/by_username?url={username}")
    if code == 404:
        return SourceResult("DEV Community", False, "not found")
    if error or data is None:
        return SourceResult("DEV Community", False, f"error: {error or 'unknown'}")
    summary = f"Articles: {data.get('articles_count', 0)}"
    return SourceResult("DEV Community", True, "found", f"https://dev.to/{username}", summary)


def check_stackoverflow(username: str) -> SourceResult:
    url = f"https://api.stackexchange.com/2.3/users?inname={username}&site=stackoverflow&pagesize=1"
    data, error, _ = fetch_json(url)
    if error or data is None:
        return SourceResult("Stack Overflow", False, f"error: {error or 'unknown'}")
    items = data.get("items", [])
    if not items:
        return SourceResult("Stack Overflow", False, "not found")
    user = items[0]
    summary = f"Reputation: {user.get('reputation', 0)}"
    return SourceResult("Stack Overflow", True, "found", user.get("link", ""), summary)


def check_dockerhub(username: str) -> SourceResult:
    data, error, code = fetch_json(f"https://hub.docker.com/v2/users/{username}/")
    if code == 404:
        return SourceResult("Docker Hub", False, "not found")
    if error or data is None:
        return SourceResult("Docker Hub", False, f"error: {error or 'unknown'}")
    summary = f"Company: {data.get('company', '') or 'n/a'}"
    return SourceResult("Docker Hub", True, "found", f"https://hub.docker.com/u/{username}", summary)


def check_bitbucket(username: str) -> SourceResult:
    data, error, code = fetch_json(f"https://api.bitbucket.org/2.0/users/{username}")
    if code == 404:
        return SourceResult("Bitbucket", False, "not found")
    if error or data is None:
        return SourceResult("Bitbucket", False, f"error: {error or 'unknown'}")
    links = data.get("links", {}).get("html", {}).get("href", "")
    return SourceResult("Bitbucket", True, "found", links, "Public account exists")


def check_chesscom(username: str) -> SourceResult:
    data, error, code = fetch_json(f"https://api.chess.com/pub/player/{username}")
    if code == 404:
        return SourceResult("Chess.com", False, "not found")
    if error or data is None:
        return SourceResult("Chess.com", False, f"error: {error or 'unknown'}")
    return SourceResult("Chess.com", True, "found", data.get("url", ""), "Public chess profile exists")


def check_pypi_search(query: str) -> SourceResult:
    url = f"https://pypi.org/search/?q={query}"
    try:
        response = requests.get(url, headers={"User-Agent": DEFAULT_UA}, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return SourceResult("PyPI search", False, f"error: {exc}")
    if response.status_code >= 400:
        return SourceResult("PyPI search", False, f"error: http {response.status_code}")
    found = "package-snippet" in response.text
    return SourceResult("PyPI search", found, "found" if found else "not found", url, "Package matches" if found else "No package matches")


def check_npm_maintainer(username: str) -> SourceResult:
    data, error, _ = fetch_json(f"https://registry.npmjs.org/-/v1/search?text=maintainer:{username}&size=1")
    if error or data is None:
        return SourceResult("npm maintainer search", False, f"error: {error or 'unknown'}")
    total = data.get("total", 0)
    found = total > 0
    return SourceResult(
        "npm maintainer search",
        found,
        "found" if found else "not found",
        f"https://www.npmjs.com/search?q=maintainer%3A{username}",
        f"Matches: {total}",
    )


def build_checks() -> List[Callable[[str], SourceResult]]:
    return [
        check_github,
        check_gitlab,
        check_reddit,
        check_hackernews,
        check_keybase,
        check_wikimedia,
        check_devto,
        check_stackoverflow,
        check_dockerhub,
        check_bitbucket,
        check_chesscom,
        check_pypi_search,
        check_npm_maintainer,
    ]


def run_checks_for_username(username: str, workers: int = 8) -> List[SourceResult]:
    checkers = build_checks()
    results: List[SourceResult] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(checkers))) as executor:
        futures = [executor.submit(checker, username) for checker in checkers]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item.source.lower())


def build_csv(rows: List[Dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["username", "source", "found", "status", "summary", "created_at", "profile_url"],
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def timeline_summary(results: List[SourceResult]) -> str:
    created = sorted([item.created_at for item in results if item.created_at])
    if not created:
        return "No public account-creation timestamps available from matched sources."
    return f"Earliest visible account creation: {created[0]} | Latest: {created[-1]}"


st.set_page_config(page_title="Public Presence Checker Pro", page_icon="🔎", layout="wide")
st.title("🔎 Public Presence Checker Pro")
st.caption("High-coverage, lawful public-source checks across developer/community platforms.")

with st.expander("Safety + legal boundary", expanded=True):
    st.write(
        "This app is for permission-based public checks only. It does NOT infer private login schedules, "
        "tracking patterns, or surveillance timelines."
    )

left, right = st.columns([2, 1])
with left:
    usernames_raw = st.text_area(
        "Usernames (comma/newline separated)",
        value="octocat",
        height=120,
        placeholder="octocat, torvalds, mozilla",
    )
with right:
    max_workers = st.slider("Parallel workers", min_value=1, max_value=16, value=8)
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
            st.info(timeline_summary(results))

            table_rows = []
            for result in results:
                row = {
                    "source": result.source,
                    "found": "yes" if result.found else "no",
                    "status": result.status,
                    "summary": result.summary,
                    "created_at": result.created_at,
                    "profile_url": result.profile_url,
                }
                table_rows.append(row)
                all_rows.append({"username": username, **row})

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
