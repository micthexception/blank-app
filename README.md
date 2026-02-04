# 🗂️ File Organizer & Index Builder

This Streamlit app lets you upload or paste text, define custom categories, build an index, and
download an organized report.

## Quickstart (local)

1. Install the requirements

   ```
   pip install -r requirements.txt
   ```

2. Run the app

   ```
   streamlit run streamlit_app.py
   ```

3. Open the app in your browser

   ```
   http://localhost:8501
   ```

## Using the app

1. Upload a file (JSON, TXT, CSV, MD, or PDF) **or** paste text into the text area.
2. Enter the categories you want to organize by (comma or newline separated).
3. Pick the detail level and whether to include all non-matching content in the Misc section.
4. Click **Organize & Build Index** to generate and download the report.

## Large files (700MB)

Streamlit uploads are limited by server memory and configuration. If large uploads fail, increase
the upload limit in a `.streamlit/config.toml` file:

```toml
[server]
maxUploadSize = 700
```

You can also paste text in chunks if needed.
