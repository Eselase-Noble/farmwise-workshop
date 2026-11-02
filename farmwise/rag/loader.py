"""Load farming guide documents from disk or the web.

Supports PDFs, Word documents, Excel sheets, CSVs, HTML and plain text/markdown
files, plus web pages and PDFs by URL (via :func:`load_url`). Each source becomes
one :class:`Document` whose ``source`` is used later for citations.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests

from farmwise import config

# Words that mark a link as worth following when crawling an org page. Kept
# agriculture-specific on purpose, so a crawl does not wander into nav pages.
RELEVANT_KEYWORDS = [
    "agric", "crop", "rice", "maize", "cereal", "farm", "harvest", "planting",
    "season", "production", "yield", "soil", "fertil", "livestock", "cocoa",
    "cassava", "irrigation", "food-security", "food security",
]

# Path/anchor fragments that mark a link as navigation, not content.
_NAV_DENY = [
    "about", "contact", "login", "signin", "sign-in", "privacy", "terms",
    "board", "management", "history", "legal", "mandate", "career", "team",
    "sitemap", "news", "media", "gallery", "faq", "staff", "vacancy", "tender",
    "cookie", "newsletter", "subscribe",
]

SUPPORTED_SUFFIXES = {
    ".pdf", ".txt", ".md",
    ".docx",                 # Word
    ".csv", ".xlsx", ".xls",  # spreadsheets
    ".html", ".htm",          # saved web pages
}


@dataclass
class Document:
    """One loaded guide: its full text and the file it came from."""

    text: str
    source: str  # file name, e.g. "sample_maize.md"


def load_file(path: Path | str) -> Document:
    """Load a supported document file into a :class:`Document`.

    Supported: .pdf, .docx, .xlsx/.xls, .csv, .html/.htm, .txt, .md.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}' for {path.name}. "
            f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    elif suffix in {".xlsx", ".xls"}:
        text = _read_spreadsheet(path, excel=True)
    elif suffix == ".csv":
        text = _read_spreadsheet(path, excel=False)
    elif suffix in {".html", ".htm"}:
        text = _html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
    else:  # .txt, .md
        text = path.read_text(encoding="utf-8", errors="ignore")

    return Document(text=text.strip(), source=path.name)


def _read_docx(path: Path) -> str:
    """Extract text from a Word .docx: paragraphs and table cells, in order."""
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _read_spreadsheet(path: Path, *, excel: bool) -> str:
    """Render a CSV or Excel workbook as readable text (one section per sheet)."""
    import pandas as pd

    def frame_to_text(df) -> str:
        df = df.fillna("")
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_csv(index=False)

    if excel:
        sheets = pd.read_excel(path, sheet_name=None)  # dict of all sheets
        blocks = []
        for name, df in sheets.items():
            blocks.append(f"## Sheet: {name}\n{frame_to_text(df)}")
        return "\n\n".join(blocks)
    return frame_to_text(pd.read_csv(path))


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF, one page after another."""
    from pypdf import PdfReader  # imported lazily so non-PDF runs don't need it

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def _read_pdf_bytes(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def _html_to_text(html: str) -> str:
    """Pull the readable text out of an HTML page.

    Uses BeautifulSoup, dropping script/style/nav/footer so we keep the article
    content rather than menus and boilerplate.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "form"]):
        tag.decompose()
    # Prefer the main/article region if the page marks one.
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")
    # Collapse the runs of blank lines that get_text leaves behind.
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def load_url(url: str, *, timeout: int = 30) -> Document:
    """Fetch a web page or PDF by URL and return it as a :class:`Document`.

    The ``source`` is the URL itself, so answers can cite where the text came
    from. HTML is reduced to readable text; PDFs are extracted page by page.
    """
    headers = {"User-Agent": "FarmWise/1.0 (workshop demo; +https://github.com/Eselase-Noble/farmwise-workshop)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()
    is_pdf = "application/pdf" in content_type or urlparse(url).path.lower().endswith(".pdf")

    if is_pdf:
        text = _read_pdf_bytes(resp.content)
    else:
        text = _html_to_text(resp.text)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError(f"No readable text extracted from {url}")
    return Document(text=text, source=url)


def discover_relevant_links(base_url: str, html: str, *, limit: int = 6) -> list[str]:
    """Find links on a page worth following: linked PDFs and same-site pages whose
    URL or anchor text looks relevant to farming. PDFs come first."""
    from bs4 import BeautifulSoup

    base_host = urlparse(base_url).netloc
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    pdfs: list[str] = []
    pages: list[str] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"]).split("#")[0]
        parsed = urlparse(href)
        if parsed.scheme not in ("http", "https") or href in seen or href == base_url:
            continue
        hay = (href + " " + (a.get_text() or "")).lower()
        is_pdf = parsed.path.lower().endswith(".pdf")
        relevant = any(k in hay for k in RELEVANT_KEYWORDS)
        is_nav = any(n in hay for n in _NAV_DENY)
        if is_pdf and not is_nav and (relevant or parsed.netloc == base_host):
            seen.add(href); pdfs.append(href)
        elif parsed.netloc == base_host and relevant and not is_nav:
            seen.add(href); pages.append(href)
    return (pdfs + pages)[:limit]


def crawl_and_load(url: str, *, max_pages: int = 5, timeout: int = 30) -> list[Document]:
    """Load a URL and, when it is an HTML page, follow its relevant links (reports,
    PDFs, same-site pages) and load those too. Returns all documents gathered.

    This is how a landing or section page (which is often a thin JavaScript shell)
    still yields real content: the actual material is in the documents it links to.
    """
    headers = {"User-Agent": "FarmWise/1.0 (workshop demo)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()
    if "application/pdf" in content_type or urlparse(url).path.lower().endswith(".pdf"):
        return [Document(text=_read_pdf_bytes(resp.content).strip(), source=url)]

    html = resp.text
    docs: list[Document] = []
    main = re.sub(r"\n{3,}", "\n\n", _html_to_text(html)).strip()
    if main:
        docs.append(Document(text=main, source=url))

    for link in discover_relevant_links(url, html, limit=max_pages):
        try:
            doc = load_url(link, timeout=timeout)
            if doc.text.strip():
                docs.append(doc)
        except Exception as exc:  # a bad link should not sink the whole crawl
            print(f"[loader] skip {link}: {exc}")
    return docs


def load_guides(directory: Optional[Path | str] = None) -> list[Document]:
    """Load every supported guide in ``directory`` (defaults to config.GUIDES_DIR).

    Skips README files. Returns documents sorted by file name for stable ordering.
    Documents that come out empty (e.g. a scanned PDF with no text layer) are
    skipped with a printed warning.
    """
    directory = Path(directory) if directory is not None else config.GUIDES_DIR
    if not directory.exists():
        raise FileNotFoundError(f"Guides directory not found: {directory}")

    docs: list[Document] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.name.lower() == "readme.md":
            continue
        doc = load_file(path)
        if not doc.text:
            print(f"[loader] WARNING: no extractable text in {path.name}, skipping")
            continue
        docs.append(doc)
    return docs
