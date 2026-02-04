import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import streamlit as st

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover - optional dependency handled in UI
    PdfReader = None


@dataclass
class CategorizationResult:
    categories: Dict[str, List[str]]
    misc: List[str]
    index: Dict[str, int]
    total_segments: int


def normalize_categories(raw: str) -> List[str]:
    categories = [cat.strip() for cat in re.split(r"[\n,]", raw) if cat.strip()]
    return list(dict.fromkeys(categories))


def split_text(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 3:
        return lines
    sentences = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", text) if seg.strip()]
    return sentences if sentences else [text]


def categorize_segments(
    segments: Iterable[str],
    categories: List[str],
    include_all_in_misc: bool,
) -> CategorizationResult:
    segment_list = list(segments)
    bucketed = {category: [] for category in categories}
    misc: List[str] = []

    for segment in segment_list:
        matched = False
        for category in categories:
            if re.search(re.escape(category), segment, re.IGNORECASE):
                bucketed[category].append(segment)
                matched = True
        if include_all_in_misc or not matched:
            misc.append(segment)

    index = {category: len(items) for category, items in bucketed.items()}
    index["Misc"] = len(misc)
    return CategorizationResult(
        categories=bucketed,
        misc=misc,
        index=index,
        total_segments=len(segment_list),
    )


def load_text_from_upload(upload) -> Tuple[str, str]:
    suffix = Path(upload.name).suffix.lower().lstrip(".")
    data = upload.read()
    if suffix in {"txt", "csv", "md"}:
        return data.decode("utf-8", errors="replace"), suffix
    if suffix == "json":
        parsed = json.loads(data.decode("utf-8", errors="replace"))
        return json.dumps(parsed, indent=2, ensure_ascii=False), suffix
    if suffix == "pdf":
        if PdfReader is None:
            raise ValueError("PyPDF2 is required to read PDF files. Please install it.")
        reader = PdfReader(upload)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages), suffix
    raise ValueError("Unsupported file type.")


def build_report(
    source_name: str,
    username: str,
    categories: List[str],
    result: CategorizationResult,
    detail_level: str,
) -> str:
    lines: List[str] = []
    lines.append("# Organized File Report")
    lines.append("")
    lines.append(f"**Source:** {source_name}")
    lines.append(f"**Prepared for:** {username}")
    lines.append("")
    lines.append("## Index")
    for category in categories:
        lines.append(f"- {category}: {result.index.get(category, 0)}")
    lines.append(f"- Misc: {result.index.get('Misc', 0)}")
    lines.append("")
    lines.append("## Categorized Content")

    for category in categories:
        lines.append("")
        lines.append(f"### {category}")
        items = result.categories.get(category, [])
        if not items:
            lines.append("- _No matches found._")
            continue
        if detail_level == "Matched snippets only":
            for item in items:
                lines.append(f"- {item}")
        else:
            lines.append("\n".join(items))

    lines.append("")
    lines.append("### Misc")
    if result.misc:
        if detail_level == "Matched snippets only":
            for item in result.misc:
                lines.append(f"- {item}")
        else:
            lines.append("\n".join(result.misc))
    else:
        lines.append("- _No misc items found._")

    return "\n".join(lines)


st.set_page_config(page_title="File Organizer", page_icon="🗂️", layout="wide")

st.title("🗂️ File Organizer & Index Builder")
st.write(
    "Upload or paste a file, define the categories to look for, and generate an indexed, "
    "organized report with a download-ready output."
)

with st.expander("Before you start"):
    st.markdown(
        """
        **Heads up on large files (700MB):** Streamlit uploads are limited by server memory and
        configuration. If a large file fails to upload, paste the content in chunks or increase
        the server's upload limits.
        """
    )

col1, col2 = st.columns(2)
with col1:
    username = st.text_input("Your name (used in the output filename)", value="user")
    categories_input = st.text_area(
        "Categories (comma or newline separated)",
        placeholder="apps, lyrics, songs, brand development, marketing, images",
        height=120,
    )
    detail_level = st.selectbox(
        "Detail level",
        ["Matched snippets only", "Full matching content"],
    )
    include_all_in_misc = st.checkbox(
        "Include all non-matching content in Misc",
        value=True,
    )

with col2:
    uploaded = st.file_uploader(
        "Upload a file (JSON, TXT, CSV, or PDF)",
        type=["json", "txt", "csv", "pdf", "md"],
    )
    pasted_text = st.text_area(
        "Or paste text here",
        placeholder="Paste your content if upload isn't working.",
        height=200,
    )

if st.button("Organize & Build Index", type="primary"):
    if not categories_input.strip():
        st.error("Please enter at least one category.")
        st.stop()

    categories = normalize_categories(categories_input)
    if not categories:
        st.error("Please enter valid categories.")
        st.stop()

    if uploaded is None and not pasted_text.strip():
        st.error("Upload a file or paste text to continue.")
        st.stop()

    try:
        if uploaded is not None:
            source_text, source_ext = load_text_from_upload(uploaded)
            source_name = uploaded.name
        else:
            source_text = pasted_text
            source_ext = "txt"
            source_name = "pasted_text.txt"
    except Exception as exc:  # pragma: no cover - UI only
        st.error(f"Could not read the file: {exc}")
        st.stop()

    segments = split_text(source_text)
    if not segments:
        st.error("No readable content found in the file.")
        st.stop()

    result = categorize_segments(segments, categories, include_all_in_misc)
    validation = categorize_segments(segments, categories, include_all_in_misc)
    validation_ok = result.index == validation.index

    report = build_report(source_name, username, categories, result, detail_level)

    st.success("Organization complete.")
    if validation_ok:
        st.info("Validation pass: counts match on a second scan.")
    else:
        st.warning("Validation pass: counts differ between scans. Please review the output.")

    st.subheader("Preview")
    st.text_area("Organized output", value=report, height=300)

    output_name = f"{Path(source_name).stem}_{username}.{source_ext if source_ext != 'pdf' else 'txt'}"
    st.download_button(
        "Download organized file",
        data=report,
        file_name=output_name,
        mime="text/plain",
    )
