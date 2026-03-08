import requests
import streamlit as st
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

REQUEST_TIMEOUT = 10
DEFAULT_UA = "public-presence-checker/1.0"


@dataclass
class SourceResult:
    source: str
    found: bool
    profile_url: Optional[str] = None
    summary: str = ""
    status: str = ""


def fetch_json(url: str, headers: Optional[Dict[str, str]] = None) -> Dict:
    merged_headers = {"User-Agent": DEFAULT_UA, **(headers or {})}
    response = requests.get(url, headers=merged_headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def check_github(username: str) -> SourceResult:
    url = f"https://api.github.com/users/{username}"
    try:
        data = fetch_json(url)
    except requests.RequestException as exc:
        return SourceResult("GitHub", False, status=f"error: {exc}")

    if data.get("message") == "Not Found":
        return SourceResult("GitHub", False, status="not found")

    summary = f"Public repos: {data.get('public_repos', 0)} | Followers: {data.get('followers', 0)}"
    return SourceResult("GitHub", True, data.get("html_url"), summary, "found")


def check_reddit(username: str) -> SourceResult:
    url = f"https://www.reddit.com/user/{username}/about.json"
    try:
        data = fetch_json(url)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return SourceResult("Reddit", False, status="not found")
        return SourceResult("Reddit", False, status=f"error: {exc}")
    except requests.RequestException as exc:
        return SourceResult("Reddit", False, status=f"error: {exc}")

    user = data.get("data", {})
    summary = f"Karma: {user.get('total_karma', 0)}"
    profile = f"https://www.reddit.com/user/{username}/"
    return SourceResult("Reddit", True, profile, summary, "found")


def check_hackernews(username: str) -> SourceResult:
    url = f"https://hacker-news.firebaseio.com/v0/user/{username}.json"
    try:
        data = fetch_json(url)
    except requests.RequestException as exc:
        return SourceResult("Hacker News", False, status=f"error: {exc}")

    if data is None:
        return SourceResult("Hacker News", False, status="not found")

    summary = f"Karma: {data.get('karma', 0)}"
    profile = f"https://news.ycombinator.com/user?id={username}"
    return SourceResult("Hacker News", True, profile, summary, "found")


def check_pypi(username: str) -> SourceResult:
    # PyPI has project endpoints, not user profile endpoints.
    # We search for projects matching the provided name.
    url = f"https://pypi.org/search/?q={username}"
    try:
        response = requests.get(url, headers={"User-Agent": DEFAULT_UA}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        return SourceResult("PyPI search", False, status=f"error: {exc}")

    found = "package-snippet" in response.text
    summary = "Potential package matches found" if found else "No package matches"
    return SourceResult("PyPI search", found, url, summary, "found" if found else "not found")


def run_checks(username: str) -> List[SourceResult]:
    checkers: List[Callable[[str], SourceResult]] = [
        check_github,
        check_reddit,
        check_hackernews,
        check_pypi,
    ]
    return [checker(username) for checker in checkers]


st.set_page_config(page_title="Public Presence Checker", page_icon="🔎", layout="centered")
st.title("🔎 Public Presence Checker")
st.caption(
    "Consent-first lookup for public developer/community profiles. "
    "Use only for lawful checks where you have permission."
)

with st.expander("Legal + ethical use", expanded=False):
    st.write(
        "This app queries only public endpoints and avoids scraping private data. "
        "Do not use it for stalking, harassment, or any non-consensual surveillance."
    )

username = st.text_input("Username to check", placeholder="octocat")
run = st.button("Run public-source check", type="primary")

if run:
    cleaned = username.strip()
    if not cleaned:
        st.error("Please enter a username.")
    else:
        with st.spinner("Checking public sources..."):
            results = run_checks(cleaned)

        found_count = sum(1 for item in results if item.found)
        st.metric("Sources with matches", f"{found_count}/{len(results)}")

        for item in results:
            emoji = "✅" if item.found else "❌"
            st.subheader(f"{emoji} {item.source}")
            st.write(f"Status: `{item.status}`")
            if item.summary:
                st.write(item.summary)
            if item.profile_url:
                st.markdown(f"[Open source result]({item.profile_url})")
