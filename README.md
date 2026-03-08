# 🔎 Public Presence Checker Pro

High-coverage Streamlit app for **lawful, public-source username checks**.

## Upgrades in this version
- More free public sources (13 total checks)
- Multi-username batch processing
- Parallel workers for faster runs
- Account-age/timeline summary from publicly exposed creation timestamps
- CSV + JSON export
- Explicit safety boundary: no private-login/surveillance inference

## Sources (public endpoints)
- GitHub
- GitLab
- Reddit
- Hacker News
- Keybase
- Wikimedia (Wikipedia)
- DEV Community
- Stack Overflow (StackExchange API)
- Docker Hub
- Bitbucket
- Chess.com
- PyPI search
- npm maintainer search

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

## Legal & ethical use
Use only with permission and lawful purpose. This tool is designed for public data checks and explicitly avoids building private behavior surveillance profiles.
