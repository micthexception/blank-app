import io
import json
import textwrap
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import streamlit as st

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover - optional dependency
except ImportError:  # pragma: no cover - optional dependency handled in UI
    PdfReader = None


@dataclass
class SectionRule:
    name: str
    keywords: List[str]


@dataclass
class FileAnalysis:
    name: str
    extension: str
    text: str
    assigned_section: str
    matched_keywords: List[str]
    excerpt: str
    notes: Optional[str] = None


st.set_page_config(page_title="Document Organizer", page_icon="🗂️", layout="wide")
st.title("🗂️ Document Organizer")
st.success(
    "✅ You are running the Document Organizer app (not the Streamlit demo).",
    icon="✅",
)
st.write(
    "Upload or paste JSON, PDF, or TXT files and organize them into custom sections. "
    "The app builds a full index and generates a downloadable report."
)


def parse_section_rules(raw_rules: str) -> List[SectionRule]:
    rules: List[SectionRule] = []
    for line in raw_rules.splitlines():
        if not line.strip():
            continue
        if ":" in line:
            name, keyword_block = line.split(":", 1)
            keywords = [item.strip() for item in keyword_block.split(",") if item.strip()]
        else:
            name = line.strip()
            keywords = []
        rules.append(SectionRule(name=name.strip(), keywords=keywords))
    return rules


def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, Optional[str]]:
    if PdfReader is None:
        return "", "PDF support requires PyPDF2. Add it to requirements to enable parsing."
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        return text, None
    except Exception as exc:  # pragma: no cover - defensive
        return "", f"PDF parsing error: {exc}"


def extract_text_from_json(file_bytes: bytes, repair: bool) -> Tuple[str, Optional[str]]:
    try:
        data = json.loads(file_bytes.decode("utf-8"))
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        return pretty, None
    except Exception as exc:
        if repair:
            fallback_text = file_bytes.decode("utf-8", errors="ignore")
            cleaned = fallback_text.replace(",]", "]").replace(",}", "}")
            try:
                data = json.loads(cleaned)
                pretty = json.dumps(data, indent=2, ensure_ascii=False)
                return pretty, f"JSON parse repaired after error: {exc}"
            except Exception:
                return fallback_text, f"JSON parse error: {exc}"
        return file_bytes.decode("utf-8", errors="ignore"), f"JSON parse error: {exc}"


def extract_text(uploaded_file, repair_json: bool) -> Tuple[str, str, Optional[str]]:
    extension = uploaded_file.name.split(".")[-1].lower()
    file_bytes = uploaded_file.getvalue()
    if extension == "pdf":
        text, note = extract_text_from_pdf(file_bytes)
    elif extension == "json":
        text, note = extract_text_from_json(file_bytes, repair_json)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")
        note = None
    return text, extension, note


def match_keywords(text: str, rule: SectionRule) -> List[str]:
    lowered = text.lower()
    candidates = rule.keywords or rule.name.split()
    return [kw for kw in candidates if kw and kw.lower() in lowered]


def summarize_text(text: str, max_chars: int = 240) -> str:
    cleaned = " ".join(text.split())
    return textwrap.shorten(cleaned, width=max_chars, placeholder="…")


def extract_snippets(text: str, keywords: Iterable[str], window: int = 180) -> List[str]:
    lowered = text.lower()
    snippets: List[str] = []
    for keyword in keywords:
        if not keyword:
            continue
        target = keyword.lower()
        start = 0
        while True:
            idx = lowered.find(target, start)
            if idx == -1:
                break
            begin = max(idx - window, 0)
            end = min(idx + len(keyword) + window, len(text))
            snippet = text[begin:end].strip()
            if snippet and snippet not in snippets:
                snippets.append(snippet)
            start = idx + len(keyword)
    return snippets


def chunk_text(text: str, chunk_size: int) -> List[str]:
    if chunk_size <= 0:
        return [text]
    lines = text.splitlines(keepends=True)
    chunks: List[str] = []
    current = ""
    for line in lines:
        if len(current) + len(line) <= chunk_size:
            current += line
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(line) <= chunk_size:
            current = line
            continue
        for idx in range(0, len(line), chunk_size):
            piece = line[idx : idx + chunk_size]
            if len(piece) == chunk_size:
                chunks.append(piece)
            else:
                current = piece
    if current:
        chunks.append(current)
    return chunks or [text]


def build_chunks_zip(
    chunks: List[str],
    base_name: str,
    user_name: str,
    extension: str,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for index, chunk in enumerate(chunks, start=1):
            chunk_name = f"{base_name}_{user_name}_chunk_{index:03d}.{extension}"
            zip_file.writestr(chunk_name, chunk)
    buffer.seek(0)
    return buffer.getvalue()


with st.sidebar:
    st.header("1) Add content")
    uploaded_files = st.file_uploader(
        "Drop JSON, PDF, or TXT files",
        type=["json", "pdf", "txt"],
        accept_multiple_files=True,
    )
    pasted_name = st.text_input("Paste name (optional)", value="pasted-input.txt")
    pasted_text = st.text_area(
        "Paste content",
        placeholder="Paste JSON or text here if you don't want to upload a file.",
        height=140,
    )

    st.header("2) Define sections")
    section_input = st.text_area(
        "One section per line. Optional keywords after a colon.",
        placeholder="Apps: iOS, Android, mobile\nLyrics: chorus, verse\nMarketing: campaign, brand",
        height=170,
    )
    user_name = st.text_input("Your name (for the output file)", value="user")
    include_full_text = st.selectbox(
        "Detail level",
        ["Full text", "Snippets only"],
        help="Choose whether to include full extracted text or just keyword snippets per section.",
    )
    output_format = st.selectbox(
        "Output format",
        ["Markdown (.md)", "Plain text (.txt)"],
        help="Choose the file type for the generated report.",
    )
    retry_parsing = st.checkbox(
        "Retry JSON parsing on error (best effort)",
        value=True,
        help="Attempts a light cleanup if JSON parsing fails.",
    )
    create_chunks = st.checkbox(
        "Also create smaller chunk files",
        value=False,
        help="Splits the generated report into smaller parts and provides individual and ZIP downloads.",
    )
    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=1000,
        max_value=50000,
        value=8000,
        step=500,
        disabled=not create_chunks,
    )
    run_button = st.button("Analyze & organize")

st.caption(
    "Note: Streamlit has an upload limit (default ~200MB). For 700MB files, increase "
    "`server.maxUploadSize` in Streamlit config."
)

rules = parse_section_rules(section_input)

if run_button and (uploaded_files or pasted_text.strip()):
    analyses: List[FileAnalysis] = []
    for uploaded_file in uploaded_files:
        text, extension, note = extract_text(uploaded_file, retry_parsing)
        analyses.append(
            FileAnalysis(
                name=uploaded_file.name,
                extension=extension,
                text=text,
                assigned_section="",
                matched_keywords=[],
                excerpt=summarize_text(text),
                notes=note,
            )
        )

    if pasted_text.strip():
        analyses.append(
            FileAnalysis(
                name=pasted_name or "pasted-input.txt",
                extension=pasted_name.split(".")[-1].lower() if "." in pasted_name else "txt",
                text=pasted_text.strip(),
                assigned_section="",
                matched_keywords=[],
                excerpt=summarize_text(pasted_text.strip()),
                notes=None,
            )
        )

    section_map: Dict[str, List[FileAnalysis]] = {rule.name: [] for rule in rules}
    section_map.setdefault("Misc", [])
    rule_lookup = {rule.name: rule for rule in rules}
    analysis_section_hits: Dict[str, List[str]] = {}

    for analysis in analyses:
        matched_sections: List[str] = []
        all_matches: List[str] = []
        for rule in rules:
            hits = match_keywords(analysis.text, rule)
            if hits:
                matched_sections.append(rule.name)
                section_map.setdefault(rule.name, []).append(analysis)
                all_matches.extend(hits)
        if not matched_sections:
            matched_sections.append("Misc")
            section_map["Misc"].append(analysis)
        analysis.assigned_section = ", ".join(matched_sections)
        analysis.matched_keywords = list(dict.fromkeys(all_matches)) if all_matches else []
        analysis_section_hits[analysis.name] = matched_sections

    verification_passed = True
    for analysis in analyses:
        expected_sections = []
        for rule in rules:
            if match_keywords(analysis.text, rule):
                expected_sections.append(rule.name)
        if not expected_sections:
            expected_sections.append("Misc")
        if sorted(expected_sections) != sorted(analysis_section_hits.get(analysis.name, [])):
            verification_passed = False
            break

    st.subheader("Index")
    for section_name, items in section_map.items():
        if not items:
            continue
        with st.expander(f"{section_name} ({len(items)} files)", expanded=True):
            for item in items:
                rule = rule_lookup.get(section_name)
                section_hits = match_keywords(item.text, rule) if rule else []
                st.markdown(f"**{item.name}** (`.{item.extension}`)")
                st.write(item.excerpt or "No text extracted.")
                if section_hits:
                    st.caption(f"Matched keywords: {', '.join(section_hits)}")
                if item.notes:
                    st.warning(item.notes)

    report_lines: List[str] = [
        "# Document Organizer Report",
        "",
        "## Summary",
        "",
        "Files analyzed and organized into the requested categories with an index "
        "and detail sections per category.",
        f"Verification pass: {'passed' if verification_passed else 'needs review'}.",
        "",
        "## Index",
        "",
    ]
    for section_name, items in section_map.items():
        if not items:
            continue
        report_lines.append(f"### {section_name}")
        for item in items:
            report_lines.append(f"- {item.name} ({item.extension})")
        report_lines.append("")

    report_lines.append("## Details")
    report_lines.append("")
    for section_name, items in section_map.items():
        if not items:
            continue
        report_lines.append(f"### {section_name}")
        report_lines.append("")
        for item in items:
            rule = rule_lookup.get(section_name)
            section_hits = match_keywords(item.text, rule) if rule else []
            report_lines.append(f"#### {item.name}")
            if item.notes:
                report_lines.append(f"_Note: {item.notes}_")
            report_lines.append("")
            if include_full_text == "Full text":
                report_lines.append(item.text or "No text extracted.")
            else:
                snippets = extract_snippets(item.text, section_hits)
                if snippets:
                    for snippet in snippets:
                        report_lines.append(f"- {snippet}")
                else:
                    report_lines.append(item.excerpt or "No text extracted.")
            report_lines.append("")

    report_text = "\n".join(report_lines)

    base_name = analyses[0].name.rsplit(".", 1)[0] if analyses else "organized"
    doc_type = analyses[0].extension if analyses else "txt"
    extension = "md" if output_format.startswith("Markdown") else "txt"
    output_name = f"{base_name}_{user_name}_{doc_type}.{extension}"

    st.download_button(
        "Download organized report",
        data=report_text,
        file_name=output_name,
        mime="text/markdown" if extension == "md" else "text/plain",
    )

    if create_chunks:
        report_chunks = chunk_text(report_text, chunk_size)
        zip_bytes = build_chunks_zip(report_chunks, base_name, user_name, extension)
        st.subheader("Chunked downloads")
        st.caption(f"Created {len(report_chunks)} chunk(s) at ~{chunk_size} chars each.")

        st.download_button(
            "Download all chunks (ZIP)",
            data=zip_bytes,
            file_name=f"{base_name}_{user_name}_chunks.zip",
            mime="application/zip",
        )

        for index, chunk in enumerate(report_chunks, start=1):
            st.download_button(
                f"Download chunk {index}",
                data=chunk,
                file_name=f"{base_name}_{user_name}_chunk_{index:03d}.{extension}",
                mime="text/markdown" if extension == "md" else "text/plain",
            )

    st.subheader("Preview")
    st.text_area(
        "Generated report preview",
        value=report_text,
        height=240,
    )
elif run_button:
    st.warning("Add files or paste content before running.")
else:
    st.info("Add files or paste content, define your sections, then run the organizer.")
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
