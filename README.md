# 🔎 Public Presence Checker Pro

A fast Streamlit app for **lawful, public-source username checks**.

## What’s upgraded
- Multi-username input (comma/newline)
- Parallel checks for speed and lower wait time
- Wider set of legal/public sources
- Downloadable CSV + JSON exports
- Clear consent-first safety messaging

## Sources queried (public endpoints)
- GitHub user API
- GitLab public user search API
- Reddit public profile endpoint
- Hacker News Firebase user endpoint
- Keybase public lookup API
- Wikimedia (Wikipedia) user query API
- DEV Community user endpoint
- PyPI search page

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tests
```bash
pytest -q
```

## Legal and ethical use
Use with permission and only for lawful, legitimate checks (e.g., your own accounts or approved security workflows). Do not use for stalking, harassment, or non-consensual surveillance.
