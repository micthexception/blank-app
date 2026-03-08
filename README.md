# 🔎 Public Presence Checker

A lightweight Streamlit app that checks **public, legal sources** for a username.

## Sources queried
- GitHub API (`/users/{username}`)
- Reddit public profile endpoint
- Hacker News Firebase user endpoint
- PyPI public search page

## Why this version
The previous script relied on many private/paid APIs and invalid endpoint patterns. This app focuses on endpoints that are publicly accessible and works with no API keys.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tests
```bash
pytest
```

## Legal and ethical use
Use only with permission and for lawful checks. Do not use for stalking, harassment, or non-consensual surveillance.
