import json
import textwrap
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import streamlit as st

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover - optional dependency
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
        reader = PdfReader(file_bytes)
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
